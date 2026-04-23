//! `NodeWorker` — rustakka child actor that executes node invocations.
//!
//! The coordinator spawns one or more `NodeWorker`s as children each
//! superstep and dispatches [`NodeInvoke`] messages to them through a
//! [`NodeDispatchRouter`]. Each invoke runs [`run_node`] (retry +
//! backoff via [`BackoffOptions`]), wraps the result in a
//! [`CoordMsg::NodeDone`], and [`pipe_to`]'s it back — no direct
//! `tokio::spawn` or hand-rolled reply wiring at the call site.
//!
//! Why actors instead of bare `tokio::spawn`?
//!
//! - Retries flow through rustakka's [`BackoffOptions`] instead of a
//!   hand-rolled `for` loop, matching how every other supervised child
//!   in the ecosystem recovers from transient failures.
//! - Reply addressing is a first-class `ActorRef<CoordMsg>` instead of
//!   the coordinator passing `self_ref().clone()` into every spawned
//!   future.
//! - A [`NodeDispatchRouter`] layer (round-robin pool for multi-target
//!   steps, single-worker for one-off) cleanly separates *how* tasks
//!   are spread from *what* each task does.
//!
//! `NodeWorker` is multi-shot: it keeps accepting `NodeInvoke` messages
//! until the parent coordinator stops it. Each invoke spawns a detached
//! task via `pipe_to` so multiple invokes can overlap — the worker's
//! `handle` returns as soon as the future is scheduled.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use rustakka_core::actor::{Actor, ActorRef, Context, Props};
use rustakka_core::pattern::{pipe_to, BackoffOptions};
use rustakka_core::routing::{BroadcastRouter, RoundRobinRouter};
use rustakka_core::supervision::{Directive, OneForOneStrategy, SupervisorStrategy};
use serde_json::Value;

use crate::context::{StoreAccessor, CURRENT_STORE};
use crate::coordinator::CoordMsg;
use crate::errors::{GraphError, GraphResult};
use crate::graph::{CacheEntry, CachePolicy, NodeCache, RetryPolicy};
use crate::node::{NodeKind, NodeOutput};
use crate::stream::{StreamWriter, CURRENT_WRITER};

/// The single message a `NodeWorker` handles. Carrying the reply address
/// (`coord`) in-band keeps the worker decoupled from any coordinator
/// global state — it only knows how to execute one node and pipe the
/// result back.
///
/// `Clone` is required by [`rustakka_core::routing::RoundRobinRouter`]
/// (it's structurally cheap because every field is already `Arc`-backed
/// or trivially copyable).
#[derive(Clone)]
pub struct NodeInvoke {
    pub task_id: String,
    pub node: String,
    pub node_kind: NodeKind,
    pub input: BTreeMap<String, Value>,
    pub writer: StreamWriter,
    pub retry: Option<RetryPolicy>,
    pub cache_policy: Option<CachePolicy>,
    pub cache: Option<Arc<NodeCache>>,
    pub store: Option<Arc<dyn StoreAccessor>>,
    pub coord: ActorRef<CoordMsg>,
}

/// Multi-shot child actor. Each `NodeInvoke` launches a detached
/// `pipe_to` future, so multiple invocations can run concurrently through
/// a single worker.
pub struct NodeWorker;

impl NodeWorker {
    pub fn new() -> Self {
        Self
    }

    /// `Props` for a node worker. A one-for-one supervisor with a
    /// [`Directive::Stop`] decider is attached so that if `handle` ever
    /// panics the worker shuts down cleanly instead of thrashing on
    /// restart. Transient *errors* — the common case — are handled
    /// inside [`run_node`] via [`BackoffOptions`] retries, and panics
    /// *inside the detached future* are caught by
    /// [`run_node_catch_unwind`] before they can escape.
    pub fn props() -> Props<NodeWorker> {
        let strategy: SupervisorStrategy =
            OneForOneStrategy::new().with_decider(|_| Directive::Stop).into();
        Props::create(NodeWorker::new).with_supervisor_strategy(strategy)
    }
}

impl Default for NodeWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for NodeWorker {
    type Msg = NodeInvoke;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: NodeInvoke) {
        let NodeInvoke {
            task_id,
            node,
            node_kind,
            input,
            writer,
            retry,
            cache_policy,
            cache,
            store,
            coord,
        } = msg;

        // Cache-hit short-circuit (policy + entry fresh) — return the
        // cached values synchronously without spawning.
        if let (Some(pol), Some(cache_ref)) = (cache_policy.as_ref(), cache.as_ref()) {
            let key = (pol.key_func)(&input);
            let now = std::time::Instant::now();
            let hit = {
                let g = cache_ref.read();
                g.get(&(node.clone(), key.clone())).and_then(|e| match e.expires_at {
                    Some(exp) if exp <= now => None,
                    _ => Some(e.value.clone()),
                })
            };
            if let Some(values) = hit {
                coord.tell(CoordMsg::NodeDone {
                    task_id,
                    node,
                    result: Ok(NodeOutput::Update(values)),
                });
                return;
            }
            // Miss: run the node, populate cache, then notify coord.
            let node_name = node.clone();
            let cache_clone = cache_ref.clone();
            let ttl = pol.ttl_seconds;
            let coord_reply = coord.clone();
            let task_id_done = task_id.clone();
            let node_done = node_name.clone();
            pipe_to(
                async move {
                    let result =
                        run_node_catch_unwind(node_kind, input, writer, retry, store).await;
                    if let Ok(NodeOutput::Update(ref values)) = result {
                        let expires_at =
                            ttl.map(|s| now + std::time::Duration::from_secs(s));
                        cache_clone.write().insert(
                            (node_name, key),
                            CacheEntry { value: values.clone(), expires_at },
                        );
                    }
                    CoordMsg::NodeDone { task_id: task_id_done, node: node_done, result }
                },
                coord_reply,
            );
            return;
        }

        // No-cache path: pipe_to the coordinator with the node result.
        let task_id_done = task_id.clone();
        let node_done = node.clone();
        pipe_to(
            async move {
                let result =
                    run_node_catch_unwind(node_kind, input, writer, retry, store).await;
                CoordMsg::NodeDone { task_id: task_id_done, node: node_done, result }
            },
            coord,
        );
    }
}

/// Dispatches [`NodeInvoke`]s across a pool of [`NodeWorker`] children.
///
/// - With a single routee, behaves exactly like `routee.tell(invoke)`.
/// - With multiple routees, load-balances via
///   [`RoundRobinRouter`] — concurrent nodes in the same superstep share
///   a small worker pool instead of each paying one-shot actor
///   setup/teardown.
/// - A [`BroadcastRouter`] variant (`broadcast_invoke`) is exposed for
///   the rare case where the *same* control message needs to reach
///   every worker (e.g. test hooks, supervisor ping).
pub struct NodeDispatchRouter {
    round_robin: RoundRobinRouter<NodeInvoke>,
    broadcast: Option<BroadcastRouter<ControlPing>>,
}

/// Idempotent control message a router may broadcast to its pool
/// (unused by the coordinator today; the type exists so
/// [`BroadcastRouter`] has a concrete `Clone` `M` to specialize on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlPing {
    Ping,
}

impl NodeDispatchRouter {
    pub fn new(routees: Vec<ActorRef<NodeInvoke>>) -> Self {
        Self { round_robin: RoundRobinRouter::new(routees), broadcast: None }
    }

    /// Install a control-broadcast channel alongside the work channel.
    /// Each `ControlPing` routee corresponds 1:1 with a work routee.
    pub fn with_broadcast(mut self, control_routees: Vec<ActorRef<ControlPing>>) -> Self {
        self.broadcast = Some(BroadcastRouter::new(control_routees));
        self
    }

    /// Fan each invoke to the next routee in round-robin order.
    pub fn route(&self, invoke: NodeInvoke) {
        self.round_robin.route(invoke);
    }

    /// Broadcast a control ping to every routee in the pool (no-op when
    /// no control channel is installed).
    pub fn ping_all(&self, ping: ControlPing) {
        if let Some(b) = &self.broadcast {
            b.route(ping);
        }
    }
}

/// Panic-safe wrapper around [`run_node`]. A panic inside a user-authored
/// node must never leave the coordinator waiting forever for a
/// `NodeDone`; we translate it into `GraphError::Other` so the superstep
/// can proceed / fail gracefully.
async fn run_node_catch_unwind(
    node_kind: NodeKind,
    input: BTreeMap<String, Value>,
    writer: StreamWriter,
    retry: Option<RetryPolicy>,
    store: Option<Arc<dyn StoreAccessor>>,
) -> GraphResult<NodeOutput> {
    match std::panic::AssertUnwindSafe(run_node(node_kind, input, writer, retry, store))
        .catch_unwind()
        .await
    {
        Ok(res) => res,
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "node panic".to_string()
            };
            Err(GraphError::other(format!("node panic: {msg}")))
        }
    }
}

/// Invoke a node honoring its optional [`RetryPolicy`]. The task-local
/// [`CURRENT_WRITER`] and (optionally) [`CURRENT_STORE`] are installed
/// around each attempt so streamed chunks / store operations remain
/// correctly attributed even across retries. Backoff delay uses
/// rustakka's [`BackoffOptions`] with a 20% jitter factor, matching the
/// semantics of every other supervised child in the runtime.
pub async fn run_node(
    node_kind: NodeKind,
    input: BTreeMap<String, Value>,
    writer: StreamWriter,
    retry: Option<RetryPolicy>,
    store: Option<Arc<dyn StoreAccessor>>,
) -> GraphResult<NodeOutput> {
    let attempts = retry.as_ref().map(|r| r.max_attempts.max(1)).unwrap_or(1);
    let backoff_ms = retry.as_ref().map(|r| r.backoff_ms).unwrap_or(0);
    let options = backoff_options(backoff_ms, attempts);
    let mut last_err: Option<GraphError> = None;

    for attempt in 0..attempts {
        let input_i = input.clone();
        let node = node_kind.clone_kind();
        let writer_i = writer.clone();
        let store_i = store.clone();
        let res = CURRENT_WRITER
            .scope(writer_i, async move {
                match store_i {
                    Some(s) => CURRENT_STORE
                        .scope(s, async move { node.invoke(input_i).await })
                        .await,
                    None => node.invoke(input_i).await,
                }
            })
            .await;
        match res {
            Ok(out) => return Ok(out),
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 < attempts && backoff_ms > 0 {
                    tokio::time::sleep(options.next_delay(attempt)).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| GraphError::other("retry policy exhausted with no error")))
}

fn backoff_options(backoff_ms: u64, attempts: u32) -> BackoffOptions {
    let min = Duration::from_millis(backoff_ms.max(1));
    // Cap the ceiling at 32x the base (or the upstream default, whichever
    // is larger) so unbounded policies don't produce multi-minute sleeps.
    let max = Duration::from_millis(backoff_ms.saturating_mul(32))
        .max(Duration::from_secs(1));
    BackoffOptions {
        min_backoff: min,
        max_backoff: max,
        random_factor: 0.2,
        max_restarts: Some(attempts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::StreamBus;
    use serde_json::json;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn run_node_retries_until_success() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_c = counter.clone();
        let node = NodeKind::from_fn(move |_| {
            let counter = counter_c.clone();
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
        });
        let writer = StreamWriter::new(StreamBus::new(), 1, "flaky");
        let retry = Some(RetryPolicy { max_attempts: 5, backoff_ms: 1 });
        let out = run_node(node, BTreeMap::new(), writer, retry, None).await.unwrap();
        match out {
            NodeOutput::Update(m) => assert_eq!(m["ok"], json!(true)),
            _ => panic!("expected Update"),
        }
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn run_node_exhausts_and_returns_error() {
        let node = NodeKind::from_fn(|_| async move {
            Err::<NodeOutput, _>(GraphError::other("boom"))
        });
        let writer = StreamWriter::new(StreamBus::new(), 1, "bad");
        let retry = Some(RetryPolicy { max_attempts: 2, backoff_ms: 0 });
        let res = run_node(node, BTreeMap::new(), writer, retry, None).await;
        assert!(matches!(res, Err(GraphError::Other(_))));
    }

    #[tokio::test]
    async fn run_node_catch_unwind_converts_panic() {
        let node = NodeKind::from_fn(|_| async move { panic!("boom in user code") });
        let writer = StreamWriter::new(StreamBus::new(), 1, "panicky");
        let res = run_node_catch_unwind(node, BTreeMap::new(), writer, None, None).await;
        match res {
            Err(GraphError::Other(msg)) => assert!(msg.contains("panic")),
            other => panic!("expected panic Err, got {other:?}"),
        }
    }

    #[test]
    fn backoff_options_jitters_within_bounds() {
        let opt = backoff_options(10, 5);
        let d0 = opt.next_delay(0);
        let d5 = opt.next_delay(5);
        assert!(d0 <= d5);
        assert!(d5 <= opt.max_backoff.mul_f64(1.0 + opt.random_factor));
    }
}
