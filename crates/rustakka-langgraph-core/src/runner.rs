//! High-level entry points: `invoke`, `stream`, `batch`.
//!
//! These hide the rustakka actor system bring-up so callers don't have to
//! manage a long-lived `ActorSystem`. For server-style usage callers can
//! reuse a coordinator by spawning it themselves.

use std::collections::BTreeMap;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use rustakka_config::Config;
use rustakka_core::actor::{ActorSystem, Props};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

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
        run_one_with_bus(graph, input, cfg, None, None, bus_for_run).await
    });
    (rx, h)
}

async fn run_one(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
) -> GraphResult<RunResult> {
    let bus = Arc::new(StreamBus::new());
    run_one_with_bus(graph, input, cfg, checkpointer, resume, bus).await
}

async fn run_one_with_bus(
    graph: Arc<CompiledStateGraph>,
    input: BTreeMap<String, Value>,
    cfg: RunnableConfig,
    checkpointer: Option<Arc<dyn CheckpointerHook>>,
    resume: Option<Value>,
    bus: Arc<StreamBus>,
) -> GraphResult<RunResult> {
    let sys = shared_system().await?;
    let topology = graph.topology().clone();
    let bus_for_actor = bus.clone();
    let cp = checkpointer.clone();
    let props = Props::create(move || {
        GraphCoordinator::new(topology.clone(), bus_for_actor.clone(), cp.clone())
    });
    let name = next_coord_name();
    let coord = sys
        .actor_of(props, &name)
        .map_err(|e| GraphError::other(e.to_string()))?;
    let (tx, rx) = oneshot::channel();
    coord.tell(CoordMsg::StartRun { input, cfg, resume, reply: tx });
    let res = rx.await.map_err(|_| GraphError::other("coordinator dropped"))?;
    coord.tell(CoordMsg::Stop);
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
}
