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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum StreamEvent {
    /// Full state after each superstep.
    Values { step: u64, values: BTreeMap<String, Value> },
    /// Per-node updates emitted at the end of the step.
    Updates { step: u64, node: String, update: BTreeMap<String, Value> },
    /// Chat-message stream (token chunks or full BaseMessage dicts).
    Messages { step: u64, node: String, message: Value },
    /// User-emitted custom payloads via `get_stream_writer()(...)`.
    Custom { step: u64, node: String, payload: Value },
    /// Verbose debug events (planning, dispatch, halt).
    Debug { step: u64, payload: Value },
}

impl StreamEvent {
    pub fn mode(&self) -> StreamMode {
        match self {
            StreamEvent::Values { .. } => StreamMode::Values,
            StreamEvent::Updates { .. } => StreamMode::Updates,
            StreamEvent::Messages { .. } => StreamMode::Messages,
            StreamEvent::Custom { .. } => StreamMode::Custom,
            StreamEvent::Debug { .. } => StreamMode::Debug,
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
}

impl StreamWriter {
    pub fn new(bus: StreamBus, step: u64, node: impl Into<String>) -> Self {
        Self { bus, step, node: node.into() }
    }

    pub fn custom(&self, payload: Value) {
        self.bus.publish(StreamEvent::Custom {
            step: self.step,
            node: self.node.clone(),
            payload,
        });
    }

    pub fn message(&self, message: Value) {
        self.bus.publish(StreamEvent::Messages {
            step: self.step,
            node: self.node.clone(),
            message,
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

        bus.publish(StreamEvent::Values { step: 1, values: BTreeMap::new() });
        bus.publish(StreamEvent::Updates {
            step: 1,
            node: "n".into(),
            update: BTreeMap::from([("x".into(), json!(1))]),
        });

        // all-mode subscriber should see two events
        assert!(matches!(all.recv().await.unwrap(), StreamEvent::Values { .. }));
        assert!(matches!(all.recv().await.unwrap(), StreamEvent::Updates { .. }));
        // updates-only subscriber should see exactly one
        assert!(matches!(only_updates.recv().await.unwrap(), StreamEvent::Updates { .. }));
    }
}
