//! High-level entry points: `invoke`, `stream`, `batch`.
//!
//! These hide the rustakka actor system bring-up so callers don't have to
//! manage a long-lived `ActorSystem`. For server-style usage callers can
//! reuse a coordinator by spawning it themselves.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::OnceCell;
use rustakka_config::Config;
use rustakka_core::actor::{ActorSystem, Props};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::config::{RunnableConfig, StreamMode};
use crate::coordinator::{CheckpointerHook, CoordMsg, GraphCoordinator, RunResult};
use crate::errors::{GraphError, GraphResult};
use crate::graph::CompiledStateGraph;
use crate::stream::{StreamBus, StreamEvent};

static SYSTEM: OnceCell<ActorSystem> = OnceCell::new();
/// Lazily create (or return) the shared `ActorSystem` for runner-spawned
/// coordinators. Callers may bring their own by using [`invoke_with_system`].
pub async fn shared_system() -> GraphResult<ActorSystem> {
    if let Some(s) = SYSTEM.get() {
        return Ok(s.clone());
    }
    let cfg = Config::reference();
    let sys = ActorSystem::create("rustakka-langgraph", cfg)
        .await
        .map_err(|e| GraphError::other(e.to_string()))?;
    let _ = SYSTEM.set(sys.clone());
    Ok(sys)
}

/// Counter for unique coordinator names within the shared system.
static COORD_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_coord_name() -> String {
    let n = COORD_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("coord-{n}-{}", uuid::Uuid::new_v4())
}

/// Run a compiled graph to completion and return the final values.
pub async fn invoke_dynamic(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
) -> GraphResult<BTreeMap<String, Value>> {
    let res = run_one(graph, input, cfg, None, None).await?;
    if let Some(intr) = res.interrupted {
        return Err(GraphError::NodeInterrupt { node: intr.node, payload: intr.value });
    }
    Ok(res.values)
}

/// Like [`invoke_dynamic`] but with an explicit checkpointer.
pub async fn invoke_with_checkpointer(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Arc<dyn CheckpointerHook>,
) -> GraphResult<RunResult> {
    run_one(graph, input, cfg, Some(checkpointer), None).await
}

/// Resume an interrupted run by supplying the resume payload.
pub async fn resume(
    graph: Arc<CompiledStateGraph>,
    cfg: RunnableConfig,
    checkpointer: Arc<dyn CheckpointerHook>,
    resume_value: Value,
) -> GraphResult<RunResult> {
    run_one(graph, BTreeMap::new(), cfg, Some(checkpointer), Some(resume_value)).await
}

/// Snapshot of a thread at a specific checkpoint, mirroring upstream's
/// `langgraph.pregel.types.StateSnapshot`. Returned from
/// [`get_state`]/[`get_state_history`]/[`update_state`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateSnapshot {
    /// Materialized `channel_name -> value` map (same shape as
    /// `RunResult.values`).
    pub values: BTreeMap<String, Value>,
    /// The superstep this snapshot was captured at.
    pub step: u64,
    /// Pending interrupt, if any — set when execution paused on this step.
    pub interrupt: Option<crate::command::Interrupt>,
    /// Config required to resume from this exact checkpoint (has
    /// `checkpoint_id` populated).
    pub config: RunnableConfig,
}

/// Fetch the latest (or specific, via `cfg.checkpoint_id`) state snapshot
/// from the attached checkpointer. Mirrors upstream
/// `CompiledStateGraph.get_state(config)`.
pub async fn get_state(
    graph: &CompiledStateGraph,
    cfg: &RunnableConfig,
    checkpointer: Arc<dyn CheckpointerHook>,
) -> GraphResult<Option<StateSnapshot>> {
    let fetched = if cfg.checkpoint_id.is_some() {
        checkpointer.get_at(cfg).await?
    } else {
        checkpointer.get_latest(cfg).await?
    };
    let Some(rep) = fetched else { return Ok(None) };
    let values = crate::state::GraphValues::new(&graph.channel_specs().to_vec());
    values.restore(rep.snapshot.clone())?;
    Ok(Some(StateSnapshot {
        values: values.snapshot_values(),
        step: rep.step,
        interrupt: rep.interrupt,
        config: cfg.clone(),
    }))
}

/// List prior state snapshots (newest first). Mirrors upstream
/// `CompiledStateGraph.get_state_history(config, limit=...)`.
pub async fn get_state_history(
    graph: &CompiledStateGraph,
    cfg: &RunnableConfig,
    checkpointer: Arc<dyn CheckpointerHook>,
    limit: Option<u32>,
) -> GraphResult<Vec<StateSnapshot>> {
    let replays = checkpointer.list_checkpoints(cfg, limit).await?;
    let mut out = Vec::with_capacity(replays.len());
    for rep in replays {
        let values = crate::state::GraphValues::new(&graph.channel_specs().to_vec());
        values.restore(rep.snapshot)?;
        out.push(StateSnapshot {
            values: values.snapshot_values(),
            step: rep.step,
            interrupt: rep.interrupt,
            config: cfg.clone(),
        });
    }
    Ok(out)
}

/// Patch state on a thread without running any nodes. Mirrors upstream
/// `CompiledStateGraph.update_state(config, values, as_node=...)`.
///
/// - Restores the latest snapshot into fresh [`GraphValues`],
/// - applies `updates` through the channel reducers,
/// - writes a new checkpoint (step += 1) labelled with `as_node` in the
///   metadata so `get_state_history` shows it as a user patch.
pub async fn update_state(
    graph: &CompiledStateGraph,
    cfg: &RunnableConfig,
    checkpointer: Arc<dyn CheckpointerHook>,
    updates: BTreeMap<String, Value>,
    as_node: Option<String>,
) -> GraphResult<RunnableConfig> {
    let mut start_step = 0u64;
    let values = crate::state::GraphValues::new(&graph.channel_specs().to_vec());
    if let Some(rep) = checkpointer.get_latest(cfg).await? {
        values.restore(rep.snapshot)?;
        start_step = rep.step;
    }
    let writes: Vec<(String, Value)> = updates.into_iter().collect();
    values.apply_writes(writes.clone())?;

    let next_step = start_step + 1;
    let pending_writes = vec![(
        as_node.unwrap_or_else(|| "<update_state>".into()),
        writes.into_iter().collect::<BTreeMap<_, _>>(),
    )];
    let snapshot = values.snapshot();
    let post = values.snapshot_values();
    checkpointer
        .put_step(cfg, next_step, &post, &snapshot, &pending_writes, None)
        .await?;

    // Return a new config pointing at the fresh checkpoint. MemorySaver-style
    // `put_step` doesn't round-trip the new id, so the easiest thing is to
    // clear `checkpoint_id` so callers always land on "latest".
    let mut new_cfg = cfg.clone();
    new_cfg.checkpoint_id = None;
    Ok(new_cfg)
}

/// Stream events as the graph executes; the future completes when the run
/// finishes. The receiver is closed when the run exits.
pub fn stream(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    modes: Vec<StreamMode>,
) -> (
    mpsc::UnboundedReceiver<StreamEvent>,
    tokio::task::JoinHandle<GraphResult<RunResult>>,
) {
    let bus = Arc::new(StreamBus::new());
    let rx = bus.subscribe(modes);
    let bus_for_run = bus.clone();
    let h = tokio::spawn(async move {
        run_one_with_bus(graph, input, cfg, None, None, bus_for_run, None).await
    });
    (rx, h)
}

/// Additive rustakka-streams entry point. Like [`stream`] but exposes
/// events as a [`rustakka_streams::Source<StreamEvent>`] wrapped in a
/// [`rustakka_streams::KillSwitch`]:
///
/// - `overflow` is `Some((size, strategy))` to bound the per-subscriber
///   buffer, or `None` for the default unbounded fan-out.
/// - Dropping the returned `KillSwitch` or calling `shutdown()` closes
///   the source and aborts the coordinator so in-flight supersteps
///   finalize cleanly.
/// - The third return value is the same `JoinHandle<GraphResult<_>>`
///   that [`stream`] returns, so existing run-completion patterns
///   transfer unchanged.
pub fn stream_source(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    modes: Vec<StreamMode>,
    overflow: Option<(usize, rustakka_streams::OverflowStrategy)>,
) -> (
    rustakka_streams::Source<StreamEvent>,
    rustakka_streams::KillSwitch,
    tokio::task::JoinHandle<GraphResult<RunResult>>,
) {
    let bus = Arc::new(StreamBus::new());
    let (subscription, source) = bus.subscribe_source(modes, overflow);
    let kill_switch = rustakka_streams::KillSwitch::new();
    let gated = kill_switch.flow(source);

    let bus_for_run = bus.clone();
    let ks_for_run = kill_switch.clone();
    let h = tokio::spawn(async move {
        // Hold the subscription for the lifetime of the run so the
        // subscriber isn't unregistered prematurely; it's dropped only
        // when the run completes or the kill-switch fires.
        let _sub = subscription;
        let res = run_one_with_bus_cancellable(
            graph,
            input,
            cfg,
            None,
            None,
            bus_for_run,
            None,
            Some(ks_for_run),
        )
        .await;
        res
    });
    (gated, kill_switch, h)
}

async fn run_one(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
) -> GraphResult<RunResult> {
    let bus = Arc::new(StreamBus::new());
    run_one_with_bus(graph, input, cfg, checkpointer, resume, bus, None).await
}

/// Run with an attached [`StoreAccessor`]. The store is made visible to
/// node bodies via [`crate::context::get_store`].
pub async fn invoke_with_store(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    store: Arc<dyn crate::context::StoreAccessor>,
) -> GraphResult<RunResult> {
    let bus = Arc::new(StreamBus::new());
    run_one_with_bus(graph, input, cfg, checkpointer, None, bus, Some(store)).await
}

/// Crate-internal: run a compiled graph and return the raw `RunResult`
/// (including `parent_command` and interrupt state). Used by the subgraph
/// adapter so parent-bound `Command`s propagate without being flattened.
pub async fn run_internal(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
) -> GraphResult<RunResult> {
    run_one(graph, input, cfg, checkpointer, resume).await
}

/// Crate-internal: like [`run_internal`] but forwards all of the child bus's
/// events (prefixed with `namespace`) to an optional parent bus so
/// `stream(subgraphs=True)` sees nested events.
pub async fn run_internal_with_parent(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
    parent: Option<(StreamBus, Vec<String>)>,
) -> GraphResult<RunResult> {
    let bus = Arc::new(StreamBus::new());
    // Express forwarding as a rustakka-streams Source → Flow(map) → Sink
    // pipeline: the child-bus receiver becomes a `Source<StreamEvent>`,
    // the `map` step injects the ancestor namespace, and
    // `Sink::for_each` writes into the parent bus. This replaces the
    // hand-rolled `while let Some(..) = rx.recv()` loop and gives us a
    // single cancellation point (the parent guard's drop, identical to
    // the old `JoinHandle::abort`).
    let forward_handle = parent.map(|(parent_bus, namespace)| {
        let rx = bus.subscribe(Vec::new());
        let source = rustakka_streams::Source::from_receiver(rx);
        let ns = namespace.clone();
        let mapped = source.map(move |mut ev| {
            let ev_ns = ev.namespace_mut();
            if ev_ns.is_empty() {
                *ev_ns = ns.clone();
            } else {
                let mut merged = ns.clone();
                merged.extend(ev_ns.iter().cloned());
                *ev_ns = merged;
            }
            ev
        });
        let parent_bus_c = parent_bus.clone();
        tokio::spawn(async move {
            rustakka_streams::Sink::for_each(mapped, move |ev| {
                parent_bus_c.publish(ev);
            })
            .await;
        })
    });
    let res = run_one_with_bus(graph, input, cfg, checkpointer, resume, bus, None).await;
    if let Some(h) = forward_handle {
        h.abort();
    }
    res
}

async fn run_one_with_bus(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
    bus: Arc<StreamBus>,
    store: Option<Arc<dyn crate::context::StoreAccessor>>,
) -> GraphResult<RunResult> {
    run_one_with_bus_cancellable(graph, input, cfg, checkpointer, resume, bus, store, None).await
}

/// Like [`run_one_with_bus`] but additionally observes a
/// [`KillSwitch`]: when it fires the in-flight coordinator receives
/// `CoordMsg::Stop` and the returned future resolves with the
/// latched coordinator reply (or a "run cancelled" error if the
/// coordinator never replied).
async fn run_one_with_bus_cancellable(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
    bus: Arc<StreamBus>,
    store: Option<Arc<dyn crate::context::StoreAccessor>>,
    kill: Option<rustakka_streams::KillSwitch>,
) -> GraphResult<RunResult> {
    let sys = shared_system().await?;
    let topology = graph.topology().clone();
    let cache = Some(graph.node_cache().clone());
    let bus_for_actor = bus.clone();
    let cp = checkpointer.clone();
    let store_for_actor = store.clone();
    let props = Props::create(move || {
        GraphCoordinator::new(topology.clone(), bus_for_actor.clone(), cp.clone())
            .with_store(store_for_actor.clone())
            .with_cache(cache.clone())
    });
    let name = next_coord_name();
    let coord = sys
        .actor_of(props, &name)
        .map_err(|e| GraphError::other(e.to_string()))?;
    // `ask_with` is rustakka's idiomatic request/reply: the closure
    // embeds the oneshot reply channel into the message, the call awaits
    // the reply or an ask timeout, and we get uniform error handling
    // (timeout vs target-dropped) for free. We pick a very long ask
    // timeout so *recursion_limit* / user-level cancellation govern run
    // duration instead of the ask pattern — the coordinator itself
    // enforces the recursion limit and streams back an error on its own.
    // Bridge an optional kill-switch → CoordMsg::Stop so dropping the
    // user-facing Source cancels an in-flight run.
    let watcher = kill.as_ref().map(|ks| {
        let coord_ref = coord.clone();
        let ks = ks.clone();
        tokio::spawn(async move {
            loop {
                if ks.is_shut_down() {
                    coord_ref.tell(CoordMsg::Stop);
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
    });

    let res = coord
        .ask_with(
            |reply| CoordMsg::StartRun { input, cfg, resume, reply },
            Duration::from_secs(60 * 60 * 24),
        )
        .await
        .map_err(|e| GraphError::other(format!("coordinator ask failed: {e}")))?;
    coord.tell(CoordMsg::Stop);
    if let Some(h) = watcher {
        h.abort();
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CompileConfig, StateGraph, END, START};
    use crate::node::{NodeKind, NodeOutput};
    use crate::state::DynamicState;
    use serde_json::json;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn end_to_end_two_node_graph() {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("a", NodeKind::from_fn(|_input| async move {
            let mut m = BTreeMap::new();
            m.insert("x".into(), json!(1));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_node("b", NodeKind::from_fn(|input| async move {
            let cur = input.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut m = BTreeMap::new();
            m.insert("x".into(), json!(cur + 41));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_edge(START, "a");
        g.add_edge("a", "b");
        g.add_edge("b", END);
        let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
        let out = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await.unwrap();
        assert_eq!(out.get("x").unwrap(), &json!(42));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recursion_limit_trips() {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("loop", NodeKind::from_fn(|_| async move {
            Ok(NodeOutput::Update(BTreeMap::new()))
        }))
        .unwrap();
        g.add_edge(START, "loop");
        g.add_edge("loop", "loop");
        let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
        let mut cfg = RunnableConfig::default();
        cfg.recursion_limit = Some(3);
        let res = invoke_dynamic(app, BTreeMap::new(), cfg).await;
        assert!(matches!(res, Err(GraphError::Recursion { limit: 3 })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compile_time_recursion_limit_used_when_config_missing() {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("loop", NodeKind::from_fn(|_| async move {
            Ok(NodeOutput::Update(BTreeMap::new()))
        }))
        .unwrap();
        g.add_edge(START, "loop");
        g.add_edge("loop", "loop");
        let mut cc = CompileConfig::default();
        cc.recursion_limit = Some(4);
        let app = Arc::new(g.compile(cc).await.unwrap());
        let res = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await;
        assert!(matches!(res, Err(GraphError::Recursion { limit: 4 })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_policy_retries_until_success() {
        use crate::graph::RetryPolicy;
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut g = StateGraph::<DynamicState>::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        g.add_node("flaky", NodeKind::from_fn(move |_| {
            let counter = counter_clone.clone();
            async move {
                let n = counter.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(GraphError::other("transient"))
                } else {
                    let mut m = BTreeMap::new();
                    m.insert("ok".into(), json!(true));
                    Ok(NodeOutput::Update(m))
                }
            }
        }))
        .unwrap();
        g.add_edge(START, "flaky");
        g.add_edge("flaky", END);
        g.set_retry_policy("flaky", RetryPolicy { max_attempts: 5, backoff_ms: 1 });

        let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
        let out = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await.unwrap();
        assert_eq!(out["ok"], json!(true));
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_policy_exhausts_and_surfaces_error() {
        use crate::graph::RetryPolicy;

        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("bad", NodeKind::from_fn(|_| async move {
            Err::<NodeOutput, _>(GraphError::other("boom"))
        }))
        .unwrap();
        g.add_edge(START, "bad");
        g.add_edge("bad", END);
        g.set_retry_policy("bad", RetryPolicy { max_attempts: 2, backoff_ms: 0 });

        let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
        let res = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await;
        assert!(matches!(res, Err(GraphError::Other(_))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interrupt_before_pauses_run() {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("a", NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("ran".into(), json!("a"));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_node("b", NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("ran".into(), json!("b"));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_edge(START, "a");
        g.add_edge("a", "b");
        g.add_edge("b", END);
        let mut cc = CompileConfig::default();
        cc.interrupt_before = vec!["b".into()];
        let app = Arc::new(g.compile(cc).await.unwrap());
        let res = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await;
        // Interrupt is surfaced as NodeInterrupt error on `invoke_dynamic`.
        assert!(matches!(res, Err(GraphError::NodeInterrupt { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interrupt_after_pauses_after_node_writes() {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("a", NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("ran".into(), json!("a"));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_node("b", NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("ran".into(), json!("b"));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_edge(START, "a");
        g.add_edge("a", "b");
        g.add_edge("b", END);
        let mut cc = CompileConfig::default();
        cc.interrupt_after = vec!["a".into()];
        let app = Arc::new(g.compile(cc).await.unwrap());
        // Use invoke_with_checkpointer so interrupts surface as a result
        // rather than being promoted to an error.
        let saver = rustakka_langgraph_checkpoint_like_noop();
        let res = invoke_with_checkpointer(
            app,
            BTreeMap::new(),
            RunnableConfig::with_thread("t-interrupt-after"),
            saver,
        )
        .await
        .unwrap();
        assert!(res.interrupted.is_some());
        // `a` wrote before the pause; `b` did not run.
        assert_eq!(res.values["ran"], json!("a"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_source_emits_events_and_completes() {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node("a", NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("x".into(), json!(7));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
        g.add_edge(START, "a");
        g.add_edge("a", END);
        let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());

        let (source, _ks, h) = stream_source(
            app,
            BTreeMap::new(),
            RunnableConfig::default(),
            vec![StreamMode::Updates, StreamMode::Values],
            None,
        );

        // Drive to completion, then collect what the source produced.
        let run_res = h.await.unwrap().unwrap();
        assert_eq!(run_res.values["x"], json!(7));

        // After the run completes the bus is dropped → source ends.
        let seen = rustakka_streams::Sink::collect(source).await;
        assert!(seen.iter().any(|e| matches!(e, StreamEvent::Updates { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kill_switch_cancels_in_flight_run() {
        let mut g = StateGraph::<DynamicState>::new();
        // Never-terminating tight loop — relies on the kill switch to stop.
        g.add_node("loop", NodeKind::from_fn(|_| async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(NodeOutput::Update(BTreeMap::new()))
        }))
        .unwrap();
        g.add_edge(START, "loop");
        g.add_edge("loop", "loop");
        let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());

        let (_source, ks, h) = stream_source(
            app,
            BTreeMap::new(),
            RunnableConfig::default(),
            vec![],
            None,
        );
        tokio::time::sleep(Duration::from_millis(80)).await;
        ks.shutdown();
        // After shutdown the run should complete within a bounded window.
        let res = tokio::time::timeout(Duration::from_secs(2), h).await;
        assert!(res.is_ok(), "run didn't terminate after kill switch");
    }

    /// Returns a no-op hook so we can use `invoke_with_checkpointer`
    /// (which surfaces interrupts) without provisioning a real saver.
    fn rustakka_langgraph_checkpoint_like_noop() -> Arc<dyn crate::coordinator::CheckpointerHook> {
        use async_trait::async_trait;

        struct Noop;
        #[async_trait]
        impl crate::coordinator::CheckpointerHook for Noop {
            async fn put_step(
                &self,
                _: &RunnableConfig,
                _: u64,
                _: &BTreeMap<String, Value>,
                _: &BTreeMap<String, crate::channel::ChannelSnapshot>,
                _: &[(String, BTreeMap<String, Value>)],
                _: Option<&crate::command::Interrupt>,
            ) -> GraphResult<()> {
                Ok(())
            }
            async fn get_latest(
                &self,
                _: &RunnableConfig,
            ) -> GraphResult<Option<crate::coordinator::CheckpointReplay>> {
                Ok(None)
            }
        }
        Arc::new(Noop)
    }
}
