//! Streaming events and the `StreamBus` actor.
//!
//! `astream` / `stream` consumers subscribe via [`StreamBus::subscribe`]; the
//! coordinator publishes [`StreamEvent`]s as supersteps progress.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
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

/// Lightweight broadcast bus. Avoids `tokio::sync::broadcast`'s lossy semantics
/// because checkpoint correctness requires every subscriber sees every event.
#[derive(Clone, Default)]
pub struct StreamBus {
    inner: Arc<StreamBusInner>,
}

#[derive(Default)]
struct StreamBusInner {
    subscribers: Mutex<Vec<Subscriber>>,
}

struct Subscriber {
    modes: Vec<StreamMode>,
    tx: mpsc::UnboundedSender<StreamEvent>,
}

impl StreamBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to a specific set of modes; pass an empty vec for "all".
    pub fn subscribe(&self, modes: Vec<StreamMode>) -> mpsc::UnboundedReceiver<StreamEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.subscribers.lock().push(Subscriber { modes, tx });
        rx
    }

    pub fn publish(&self, ev: StreamEvent) {
        let mode = ev.mode();
        let mut subs = self.inner.subscribers.lock();
        subs.retain(|s| {
            if !s.modes.is_empty() && !s.modes.contains(&mode) {
                return true;
            }
            s.tx.send(ev.clone()).is_ok()
        });
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
}
