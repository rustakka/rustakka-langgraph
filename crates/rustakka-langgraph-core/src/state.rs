//! `GraphState` schema trait + dynamic `GraphValues` runtime container.
//!
//! Two paths coexist:
//!  - **Typed Rust**: implement `GraphState` (or use `#[derive(GraphState)]`)
//!    on a `serde`-compatible struct. Field annotations choose reducers.
//!  - **Dynamic / Python**: build a `GraphValues` directly from a list of
//!    `ChannelSpec`s; the Python shim uses this path so user-defined
//!    `TypedDict`s never need code-gen.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::channel::{Channel, ChannelKind, ChannelSnapshot, ChannelValue, StoredChannel};
use crate::errors::{GraphError, GraphResult};

/// One channel declared on a state schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSpec {
    pub name: String,
    /// Reducer name (`last_value`, `topic`, `add_messages`, `add`, `extend`, ...).
    /// Resolved via [`ChannelKind::from_name`].
    pub reducer: String,
}

impl ChannelSpec {
    pub fn last_value(name: impl Into<String>) -> Self {
        Self { name: name.into(), reducer: "last_value".into() }
    }
    pub fn add_messages(name: impl Into<String>) -> Self {
        Self { name: name.into(), reducer: "add_messages".into() }
    }
    pub fn topic(name: impl Into<String>) -> Self {
        Self { name: name.into(), reducer: "topic".into() }
    }
    pub fn kind(&self) -> ChannelKind {
        ChannelKind::from_name(&self.reducer)
    }
}

/// Implemented by typed Rust state schemas.
pub trait GraphState: Send + Sync + 'static {
    fn channel_specs() -> Vec<ChannelSpec>;
    fn to_values(&self) -> BTreeMap<String, Value>;
    fn from_values(values: &BTreeMap<String, Value>) -> GraphResult<Self>
    where
        Self: Sized + serde::de::DeserializeOwned,
    {
        let v = serde_json::to_value(values).map_err(|e| GraphError::other(e.to_string()))?;
        serde_json::from_value(v).map_err(|e| GraphError::other(e.to_string()))
    }
}

/// Trivial schema for fully-dynamic graphs: every key is `LastValue`.
/// Used when callers don't want to declare a schema upfront (Python default).
#[derive(Debug, Clone, Default)]
pub struct DynamicState;

impl GraphState for DynamicState {
    fn channel_specs() -> Vec<ChannelSpec> {
        Vec::new()
    }
    fn to_values(&self) -> BTreeMap<String, Value> {
        BTreeMap::new()
    }
}

// ---------------------------- runtime container ----------------------------

/// Live channel container shared across the graph runtime.
#[derive(Debug, Clone)]
pub struct GraphValues {
    inner: Arc<GraphValuesInner>,
}

#[derive(Debug)]
struct GraphValuesInner {
    channels: dashmap::DashMap<String, Arc<dyn Channel>>,
    /// Insertion order (stable for serialization).
    order: parking_lot::RwLock<Vec<String>>,
}

impl GraphValues {
    pub fn new(specs: &[ChannelSpec]) -> Self {
        let me = Self {
            inner: Arc::new(GraphValuesInner {
                channels: dashmap::DashMap::new(),
                order: parking_lot::RwLock::new(Vec::new()),
            }),
        };
        for s in specs {
            me.ensure_channel(&s.name, s.kind());
        }
        me
    }

    /// Lazily declare a channel; reducer defaults to `LastValue` if unknown.
    ///
    /// The previous implementation used `contains_key` + `insert`, which
    /// races under concurrent fan-out (two dispatch targets touching the same
    /// channel name can both observe `!contains_key` and then both `insert`,
    /// leaking out-of-order entries into `order`). Using `DashMap::entry`
    /// collapses the check-and-set into a single atomic guard and we only
    /// push into `order` on the branch that actually created the entry.
    pub fn ensure_channel(&self, name: &str, kind: ChannelKind) {
        let mut inserted = false;
        self.inner.channels.entry(name.to_string()).or_insert_with(|| {
            inserted = true;
            StoredChannel::new(kind)
        });
        if inserted {
            self.inner.order.write().push(name.into());
        }
    }

    pub fn channel(&self, name: &str) -> Option<Arc<dyn Channel>> {
        self.inner.channels.get(name).map(|r| r.clone())
    }

    pub fn channel_names(&self) -> Vec<String> {
        self.inner.order.read().clone()
    }

    /// Returns names of channels that received at least one update since the
    /// last `ack_planned` call.
    pub fn updated_channels(&self) -> Vec<String> {
        self.inner
            .channels
            .iter()
            .filter_map(|kv| if kv.value().updated() { Some(kv.key().clone()) } else { None })
            .collect()
    }

    pub fn ack_all_planned(&self) {
        for kv in self.inner.channels.iter() {
            kv.value().ack_planned();
        }
    }

    pub fn begin_step(&self) {
        for kv in self.inner.channels.iter() {
            kv.value().begin_step();
        }
    }

    /// Apply input values as the very first set of channel updates.
    pub fn seed(&self, input: BTreeMap<String, Value>) -> GraphResult<()> {
        for (k, v) in input {
            self.ensure_channel(&k, ChannelKind::LastValue);
            let ch = self.channel(&k).expect("just-ensured channel");
            ch.apply(vec![v])?;
        }
        Ok(())
    }

    /// Apply a batch of `(channel, update)` pairs from a node.
    pub fn apply_writes(&self, writes: Vec<(String, ChannelValue)>) -> GraphResult<()> {
        // Group by channel so reducers see a single batch per channel.
        let mut grouped: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        for (c, v) in writes {
            grouped.entry(c).or_default().push(v);
        }
        for (name, updates) in grouped {
            self.ensure_channel(&name, ChannelKind::LastValue);
            let ch = self.channel(&name).expect("just-ensured channel");
            ch.apply(updates)?;
        }
        Ok(())
    }

    /// Materialize the current values as a plain map (for callers / serializers).
    pub fn snapshot_values(&self) -> BTreeMap<String, Value> {
        let mut out = BTreeMap::new();
        for name in self.channel_names() {
            if let Some(ch) = self.channel(&name) {
                if let Some(v) = ch.get() {
                    out.insert(name, v);
                }
            }
        }
        out
    }

    /// Snapshot every channel for checkpoint storage.
    pub fn snapshot(&self) -> BTreeMap<String, ChannelSnapshot> {
        let mut out = BTreeMap::new();
        for name in self.channel_names() {
            if let Some(ch) = self.channel(&name) {
                out.insert(name, ch.snapshot());
            }
        }
        out
    }

    pub fn restore(&self, snaps: BTreeMap<String, ChannelSnapshot>) -> GraphResult<()> {
        for (name, snap) in snaps {
            self.ensure_channel(&name, snap.kind.clone());
            let ch = self.channel(&name).expect("just-ensured channel");
            ch.restore(snap)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn seed_and_snapshot() {
        let v = GraphValues::new(&[ChannelSpec::last_value("x")]);
        let mut input = BTreeMap::new();
        input.insert("x".into(), json!(42));
        v.seed(input).unwrap();
        assert_eq!(v.snapshot_values()["x"], json!(42));
        assert!(v.updated_channels().contains(&"x".to_string()));
        v.ack_all_planned();
        assert!(v.updated_channels().is_empty());
    }

    #[test]
    fn apply_writes_groups_per_channel() {
        let v = GraphValues::new(&[
            ChannelSpec { name: "msgs".into(), reducer: "topic".into() },
        ]);
        v.apply_writes(vec![
            ("msgs".into(), json!("a")),
            ("msgs".into(), json!("b")),
        ])
        .unwrap();
        assert_eq!(v.snapshot_values()["msgs"], json!(["a", "b"]));
    }
}
