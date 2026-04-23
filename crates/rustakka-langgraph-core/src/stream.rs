//! Streaming events and the `StreamBus` actor.
//!
//! `astream` / `stream` consumers subscribe via [`StreamBus::subscribe`]; the
//! coordinator publishes [`StreamEvent`]s as supersteps progress.
//!
//! Internally the bus is implemented as a *message-driven actor*: a
//! dedicated tokio task owns the subscriber list (no mutex), and
//! [`StreamBus`] holds an `UnboundedSender<BusCmd>` to that task.
//! Callers interact only through [`publish`], [`subscribe`] and
//! [`subscribe_source`]; the actor shape keeps the bus consistent with
//! the rest of the runtime even though it's not spawned through
//! `ActorSystem` (the bus outlives any single coordinator).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rustakka_streams::{OverflowStrategy, Source};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::config::StreamMode;

/// Upstream `langgraph` mirrors stream events as plain dicts; we keep them
/// strongly typed and serializable.
///
/// The `namespace` field is `Vec<String>` (empty for root graph, populated
/// with the ancestor node path when emitted from a subgraph under
/// `stream(..., subgraphs=True)`). It's a separate field rather than a
/// variant because every event is equally nestable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Full state after each superstep.
    Values {
        step: u64,
        values: BTreeMap<String, Value>,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Per-node updates emitted at the end of the step.
    Updates {
        step: u64,
        node: String,
        update: BTreeMap<String, Value>,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Chat-message stream (token chunks or full BaseMessage dicts).
    Messages {
        step: u64,
        node: String,
        message: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// User-emitted custom payloads via `get_stream_writer()(...)`.
    Custom {
        step: u64,
        node: String,
        payload: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Verbose debug events (planning, dispatch, halt).
    Debug {
        step: u64,
        payload: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
    // ----- astream_events v2 event kinds -----
    /// Fired when a node starts executing. Mirrors `on_chain_start`.
    OnChainStart {
        step: u64,
        node: String,
        run_id: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Fired after a node emits its writes. Mirrors `on_chain_end`.
    OnChainEnd {
        step: u64,
        node: String,
        run_id: Option<String>,
        output: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Token-level chunks from an LLM. Mirrors `on_chat_model_stream`.
    OnChatModelStream {
        step: u64,
        node: String,
        chunk: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Fired when a tool starts executing. Mirrors `on_tool_start`.
    OnToolStart {
        step: u64,
        node: String,
        tool: String,
        input: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
    /// Fired when a tool finishes. Mirrors `on_tool_end`.
    OnToolEnd {
        step: u64,
        node: String,
        tool: String,
        output: Value,
        #[serde(default)]
        namespace: Vec<String>,
    },
}

impl StreamEvent {
    pub fn mode(&self) -> StreamMode {
        match self {
            StreamEvent::Values { .. } => StreamMode::Values,
            StreamEvent::Updates { .. } => StreamMode::Updates,
            StreamEvent::Messages { .. } => StreamMode::Messages,
            StreamEvent::Custom { .. } => StreamMode::Custom,
            StreamEvent::Debug { .. } => StreamMode::Debug,
            // All v2 events map onto the "events" mode, which is distinct
            // from the classic `stream_mode` strings.
            StreamEvent::OnChainStart { .. }
            | StreamEvent::OnChainEnd { .. }
            | StreamEvent::OnChatModelStream { .. }
            | StreamEvent::OnToolStart { .. }
            | StreamEvent::OnToolEnd { .. } => StreamMode::Events,
        }
    }

    /// Event-kind string matching upstream's Python strings
    /// (`"values" | "updates" | "messages" | "custom" | "debug" | "events"`).
    pub fn mode_string(&self) -> &'static str {
        match self {
            StreamEvent::Values { .. } => "values",
            StreamEvent::Updates { .. } => "updates",
            StreamEvent::Messages { .. } => "messages",
            StreamEvent::Custom { .. } => "custom",
            StreamEvent::Debug { .. } => "debug",
            StreamEvent::OnChainStart { .. }
            | StreamEvent::OnChainEnd { .. }
            | StreamEvent::OnChatModelStream { .. }
            | StreamEvent::OnToolStart { .. }
            | StreamEvent::OnToolEnd { .. } => "events",
        }
    }

    /// Mutable access to the namespace. Used when forwarding events across
    /// subgraph boundaries so the parent bus sees `(ancestor_node, ...)`.
    pub fn namespace_mut(&mut self) -> &mut Vec<String> {
        match self {
            StreamEvent::Values { namespace, .. }
            | StreamEvent::Updates { namespace, .. }
            | StreamEvent::Messages { namespace, .. }
            | StreamEvent::Custom { namespace, .. }
            | StreamEvent::Debug { namespace, .. }
            | StreamEvent::OnChainStart { namespace, .. }
            | StreamEvent::OnChainEnd { namespace, .. }
            | StreamEvent::OnChatModelStream { namespace, .. }
            | StreamEvent::OnToolStart { namespace, .. }
            | StreamEvent::OnToolEnd { namespace, .. } => namespace,
        }
    }
}

/// Opaque subscriber identifier. Returned by
/// [`StreamBus::subscribe_with_id`] and used with
/// [`StreamBus::unsubscribe`].
pub type SubId = u64;

/// Broadcast bus for [`StreamEvent`]s. Cheap to [`Clone`] — all clones
/// share the same underlying subscriber registry.
///
/// The bus is intentionally *actor-shaped*: all mutations of the
/// subscriber list (`subscribe`, `unsubscribe`) are serialized through
/// a single writer lock so concurrent runs can't observe a torn view.
/// `publish` takes a read lock and fans out inline — this keeps strict
/// happens-before ordering between "the coordinator emitted X" and
/// "subscribers have X in their mpsc" without a task-hop, which the
/// existing streaming tests rely on. The alternative (a separate bus
/// task forwarding BusCmd::Publish) would make `publish` asynchronous
/// with respect to the coordinator's visible timeline and break the
/// post-`run.await` drain contract.
///
/// The bus avoids `tokio::sync::broadcast`'s lossy semantics because
/// checkpoint correctness requires every subscriber sees every event.
#[derive(Clone)]
pub struct StreamBus {
    subs: Arc<parking_lot::RwLock<Vec<Subscriber>>>,
    next_id: Arc<AtomicU64>,
}

impl Default for StreamBus {
    fn default() -> Self {
        Self::new()
    }
}

struct Subscriber {
    id: SubId,
    modes: Vec<StreamMode>,
    tx: mpsc::UnboundedSender<StreamEvent>,
}

impl StreamBus {
    /// Construct a new bus.
    pub fn new() -> Self {
        Self {
            subs: Arc::new(parking_lot::RwLock::new(Vec::new())),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    fn alloc_id(&self) -> SubId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Subscribe to a specific set of modes; pass an empty vec for "all".
    pub fn subscribe(&self, modes: Vec<StreamMode>) -> mpsc::UnboundedReceiver<StreamEvent> {
        let (_, rx) = self.subscribe_with_id(modes);
        rx
    }

    /// Like [`Self::subscribe`] but also returns the [`SubId`] so the
    /// caller can later [`Self::unsubscribe`]. Used by
    /// [`subscribe_source`] to wire up a `KillSwitch` that terminates
    /// *just this subscriber*.
    pub fn subscribe_with_id(
        &self,
        modes: Vec<StreamMode>,
    ) -> (SubId, mpsc::UnboundedReceiver<StreamEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let id = self.alloc_id();
        self.subs.write().push(Subscriber { id, modes, tx });
        (id, rx)
    }

    /// Remove a subscriber previously returned by
    /// [`subscribe_with_id`]. No-op if already gone.
    pub fn unsubscribe(&self, id: SubId) {
        self.subs.write().retain(|s| s.id != id);
    }

    /// Broadcast an event. Fire-and-forget — matches the pre-actor
    /// signature. Dead subscribers are garbage-collected lazily: a
    /// failed send triggers an upgrade to a write lock which removes
    /// closed channels, so slow consumers that `drop` their receiver
    /// don't leak.
    pub fn publish(&self, ev: StreamEvent) {
        let mode = ev.mode();
        let mut dead: Vec<SubId> = Vec::new();
        {
            let guard = self.subs.read();
            for s in guard.iter() {
                if !s.modes.is_empty() && !s.modes.contains(&mode) {
                    continue;
                }
                if s.tx.send(ev.clone()).is_err() {
                    dead.push(s.id);
                }
            }
        }
        if !dead.is_empty() {
            let mut w = self.subs.write();
            w.retain(|s| !dead.contains(&s.id));
        }
    }

    /// Subscribe and materialize the stream as a rustakka-streams
    /// [`Source`]. `overflow` is applied only when `buffer_size` is
    /// `Some(n)`; otherwise the bus delivers with unbounded fan-out
    /// (matching the legacy [`subscribe`] semantics exactly).
    ///
    /// Returns `(subscription, source)`. Dropping the returned
    /// [`Subscription`] unregisters the subscriber from the bus — the
    /// attached source then sees end-of-stream. This is the preferred
    /// subscription API from within Rust; Python and legacy Rust
    /// callers stay on the [`subscribe`]/[`subscribe_with_id`] pair.
    pub fn subscribe_source(
        &self,
        modes: Vec<StreamMode>,
        buffer: Option<(usize, OverflowStrategy)>,
    ) -> (Subscription, Source<StreamEvent>) {
        let (id, rx) = self.subscribe_with_id(modes);
        let src = Source::from_receiver(rx);
        let src = match buffer {
            Some((size, strategy)) => src.buffer(size, strategy),
            None => src,
        };
        (Subscription { bus: self.clone(), id: Some(id) }, src)
    }
}

/// RAII-style handle that unregisters its bus subscription when dropped.
/// See [`StreamBus::subscribe_source`].
pub struct Subscription {
    bus: StreamBus,
    id: Option<SubId>,
}

impl Subscription {
    /// Unsubscribe explicitly; equivalent to dropping.
    pub fn unsubscribe(mut self) {
        if let Some(id) = self.id.take() {
            self.bus.unsubscribe(id);
        }
    }

    pub fn id(&self) -> Option<SubId> {
        self.id
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            self.bus.unsubscribe(id);
        }
    }
}

/// Handle installed into a node's task-local context so the node body can emit
/// `Custom`/`Messages` stream events. Mirrors upstream's `get_stream_writer()`.
#[derive(Clone)]
pub struct StreamWriter {
    bus: StreamBus,
    step: u64,
    node: String,
    /// Subgraph ancestor path; empty for the root graph.
    namespace: Vec<String>,
}

impl StreamWriter {
    pub fn new(bus: StreamBus, step: u64, node: impl Into<String>) -> Self {
        Self { bus, step, node: node.into(), namespace: Vec::new() }
    }

    pub fn with_namespace(mut self, ns: Vec<String>) -> Self {
        self.namespace = ns;
        self
    }

    /// Borrow the underlying bus (used by the subgraph adapter to forward
    /// events to the parent run).
    pub fn bus(&self) -> &StreamBus {
        &self.bus
    }

    pub fn namespace(&self) -> &[String] {
        &self.namespace
    }

    pub fn node(&self) -> &str {
        &self.node
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn custom(&self, payload: Value) {
        self.bus.publish(StreamEvent::Custom {
            step: self.step,
            node: self.node.clone(),
            payload,
            namespace: self.namespace.clone(),
        });
    }

    pub fn message(&self, message: Value) {
        self.bus.publish(StreamEvent::Messages {
            step: self.step,
            node: self.node.clone(),
            message,
            namespace: self.namespace.clone(),
        });
    }

    pub fn chat_model_chunk(&self, chunk: Value) {
        self.bus.publish(StreamEvent::OnChatModelStream {
            step: self.step,
            node: self.node.clone(),
            chunk,
            namespace: self.namespace.clone(),
        });
    }

    pub fn tool_start(&self, tool: impl Into<String>, input: Value) {
        self.bus.publish(StreamEvent::OnToolStart {
            step: self.step,
            node: self.node.clone(),
            tool: tool.into(),
            input,
            namespace: self.namespace.clone(),
        });
    }

    pub fn tool_end(&self, tool: impl Into<String>, output: Value) {
        self.bus.publish(StreamEvent::OnToolEnd {
            step: self.step,
            node: self.node.clone(),
            tool: tool.into(),
            output,
            namespace: self.namespace.clone(),
        });
    }

    pub fn chain_start(&self, run_id: Option<String>, tags: Vec<String>) {
        self.bus.publish(StreamEvent::OnChainStart {
            step: self.step,
            node: self.node.clone(),
            run_id,
            tags,
            namespace: self.namespace.clone(),
        });
    }

    pub fn chain_end(&self, run_id: Option<String>, output: Value) {
        self.bus.publish(StreamEvent::OnChainEnd {
            step: self.step,
            node: self.node.clone(),
            run_id,
            output,
            namespace: self.namespace.clone(),
        });
    }
}

tokio::task_local! {
    pub static CURRENT_WRITER: StreamWriter;
}

/// Convenience: access the currently installed writer, if any. Returns `None`
/// outside of a node body.
pub fn current_writer() -> Option<StreamWriter> {
    CURRENT_WRITER.try_with(|w| w.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn publishes_to_matching_subscribers() {
        let bus = StreamBus::new();
        let mut all = bus.subscribe(vec![]);
        let mut only_updates = bus.subscribe(vec![StreamMode::Updates]);

        bus.publish(StreamEvent::Values {
            step: 1,
            values: BTreeMap::new(),
            namespace: Vec::new(),
        });
        bus.publish(StreamEvent::Updates {
            step: 1,
            node: "n".into(),
            update: BTreeMap::from([("x".into(), json!(1))]),
            namespace: Vec::new(),
        });

        // all-mode subscriber should see two events
        assert!(matches!(all.recv().await.unwrap(), StreamEvent::Values { .. }));
        assert!(matches!(all.recv().await.unwrap(), StreamEvent::Updates { .. }));
        // updates-only subscriber should see exactly one
        assert!(matches!(only_updates.recv().await.unwrap(), StreamEvent::Updates { .. }));
    }

    #[tokio::test]
    async fn unsubscribe_stops_delivery() {
        let bus = StreamBus::new();
        let (id, mut rx) = bus.subscribe_with_id(vec![]);
        bus.publish(StreamEvent::Debug {
            step: 0,
            payload: json!("a"),
            namespace: Vec::new(),
        });
        // Allow the bus actor to process the Publish before we unsubscribe.
        assert!(matches!(rx.recv().await.unwrap(), StreamEvent::Debug { .. }));
        bus.unsubscribe(id);
        // Give the bus actor a chance to process the Unsubscribe, then
        // publish — the subscriber should no longer receive.
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        bus.publish(StreamEvent::Debug {
            step: 0,
            payload: json!("b"),
            namespace: Vec::new(),
        });
        // After unsubscribe the bus drops its tx for this subscriber,
        // which closes the channel. `rx.recv()` therefore returns
        // `None` (not a spurious event).
        let closed = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            rx.recv(),
        )
        .await;
        assert!(matches!(closed, Ok(None)), "expected closed channel, got {:?}", closed);
    }

    /// `subscribe_source` returns a `Source<StreamEvent>` that dries up
    /// as soon as the accompanying `Subscription` is dropped; it's the
    /// Source-native pair of the legacy `subscribe_with_id` / `unsubscribe`
    /// pair.
    #[tokio::test]
    async fn subscribe_source_delivers_then_closes_on_drop() {
        let bus = StreamBus::new();
        let (sub, source) = bus.subscribe_source(vec![], None);
        bus.publish(StreamEvent::Debug {
            step: 0,
            payload: json!("first"),
            namespace: Vec::new(),
        });
        bus.publish(StreamEvent::Debug {
            step: 1,
            payload: json!("second"),
            namespace: Vec::new(),
        });
        drop(sub);
        let collected: Vec<StreamEvent> = rustakka_streams::Sink::collect(source).await;
        assert!(collected.len() >= 1, "expected at least one event, got {}", collected.len());
        assert!(collected.iter().all(|e| matches!(e, StreamEvent::Debug { .. })));
    }

    /// With `OverflowStrategy::DropHead` and a tiny buffer, a backlogged
    /// subscriber must never block the bus (so `publish` returns promptly)
    /// and delivers at most `size` buffered items.
    #[tokio::test]
    async fn subscribe_source_overflow_drop_head_bounds_buffer() {
        let bus = StreamBus::new();
        let (_sub, source) =
            bus.subscribe_source(vec![], Some((2, OverflowStrategy::DropHead)));
        for i in 0..20u64 {
            bus.publish(StreamEvent::Debug {
                step: i,
                payload: json!(i),
                namespace: Vec::new(),
            });
        }
        drop(_sub);
        let collected: Vec<StreamEvent> = rustakka_streams::Sink::collect(source).await;
        // A bounded buffer of size 2 + DropHead cannot possibly retain all
        // 20 events; the exact count depends on scheduling but must be
        // strictly less than what we published.
        assert!(collected.len() < 20, "overflow did not bound buffer: {}", collected.len());
    }
}
