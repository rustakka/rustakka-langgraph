//! Typed communication channels.
//!
//! Mirrors LangGraph's `langgraph.channels.*`:
//! - `LastValue`              — overwrite-with-latest
//! - `Topic`                  — accumulating PubSub queue
//! - `BinaryOperatorAggregate`— fold updates with a binary op
//! - `EphemeralValue`         — single-superstep visibility, then cleared
//! - `AnyValue`               — store an arbitrary `Value` (no constraint)
//!
//! For the pure-Rust path we keep values as `serde_json::Value` so we can
//! interop transparently with the Python shim and `serde`-based checkpoints.
//! Strongly-typed channels can be wrapped on top via `ChannelValue::cast`.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::errors::{GraphError, GraphResult};

/// Discriminator for which reducer a channel uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ChannelKind {
    LastValue,
    Topic { unique: bool, accumulate: bool },
    BinaryOperator { op: BinaryOp },
    Ephemeral,
    AnyValue,
    /// Reducer for chat messages: `add_messages(left, right)`.
    AddMessages,
}

impl ChannelKind {
    pub fn from_name(name: &str) -> ChannelKind {
        match name {
            "topic"        => ChannelKind::Topic { unique: false, accumulate: true },
            "ephemeral"    => ChannelKind::Ephemeral,
            "any"|"any_value" => ChannelKind::AnyValue,
            "add"|"add_value" => ChannelKind::BinaryOperator { op: BinaryOp::Add },
            "extend"|"concat" => ChannelKind::BinaryOperator { op: BinaryOp::Extend },
            "merge"|"merge_dicts" => ChannelKind::BinaryOperator { op: BinaryOp::MergeDicts },
            "add_messages" | "messages" => ChannelKind::AddMessages,
            _ => ChannelKind::LastValue,
        }
    }
}

/// Built-in binary operators used by `BinaryOperatorAggregate`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    Add,
    Extend,
    MergeDicts,
}

/// A boxed channel value (JSON for portability).
pub type ChannelValue = Value;

/// Trait every channel implements.
pub trait Channel: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> &ChannelKind;
    /// Apply pending updates and produce the post-superstep stored value.
    /// `updates` is moved (drained) into the reducer.
    fn apply(&self, updates: Vec<ChannelValue>) -> GraphResult<()>;
    /// Read the channel's current value (if any).
    fn get(&self) -> Option<ChannelValue>;
    /// True if the channel has ever received an update (used for planning).
    fn updated(&self) -> bool;
    /// Reset the "updated since last plan" flag.
    fn ack_planned(&self);
    /// Snapshot the channel for checkpointing.
    fn snapshot(&self) -> ChannelSnapshot;
    /// Restore from a snapshot.
    fn restore(&self, snap: ChannelSnapshot) -> GraphResult<()>;
    /// At the start of a new superstep, ephemeral channels clear themselves.
    fn begin_step(&self);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSnapshot {
    pub kind: ChannelKind,
    pub value: Option<ChannelValue>,
    pub updated: bool,
}

// ------------------------ Concrete channel impls ------------------------

#[derive(Debug)]
pub struct StoredChannel {
    kind: ChannelKind,
    inner: RwLock<ChannelInner>,
}

#[derive(Debug)]
struct ChannelInner {
    value: Option<ChannelValue>,
    updated: bool,
}

impl StoredChannel {
    pub fn new(kind: ChannelKind) -> Arc<dyn Channel> {
        Arc::new(StoredChannel {
            kind,
            inner: RwLock::new(ChannelInner { value: None, updated: false }),
        })
    }

    pub fn from_snapshot(snap: ChannelSnapshot) -> Arc<dyn Channel> {
        Arc::new(StoredChannel {
            kind: snap.kind,
            inner: RwLock::new(ChannelInner { value: snap.value, updated: snap.updated }),
        })
    }
}

impl Channel for StoredChannel {
    fn kind(&self) -> &ChannelKind {
        &self.kind
    }

    fn apply(&self, updates: Vec<ChannelValue>) -> GraphResult<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut inner = self.inner.write();
        match &self.kind {
            ChannelKind::LastValue => {
                inner.value = updates.into_iter().last();
            }
            ChannelKind::Ephemeral => {
                inner.value = updates.into_iter().last();
            }
            ChannelKind::AnyValue => {
                inner.value = updates.into_iter().last();
            }
            ChannelKind::Topic { unique, accumulate } => {
                let mut items: Vec<Value> = match (accumulate, inner.value.take()) {
                    (true, Some(Value::Array(a))) => a,
                    _ => Vec::new(),
                };
                for u in updates {
                    match u {
                        Value::Array(arr) => items.extend(arr),
                        other => items.push(other),
                    }
                }
                if *unique {
                    let mut seen = std::collections::HashSet::new();
                    items.retain(|v| seen.insert(v.to_string()));
                }
                inner.value = Some(Value::Array(items));
            }
            ChannelKind::BinaryOperator { op } => {
                let mut acc = inner.value.take();
                for u in updates {
                    acc = Some(reduce_binary(op, acc, u)?);
                }
                inner.value = acc;
            }
            ChannelKind::AddMessages => {
                let mut left = match inner.value.take() {
                    Some(Value::Array(a)) => a,
                    Some(other) => vec![other],
                    None => Vec::new(),
                };
                for u in updates {
                    add_messages_into(&mut left, u);
                }
                inner.value = Some(Value::Array(left));
            }
        }
        inner.updated = true;
        Ok(())
    }

    fn get(&self) -> Option<ChannelValue> {
        self.inner.read().value.clone()
    }

    fn updated(&self) -> bool {
        self.inner.read().updated
    }

    fn ack_planned(&self) {
        self.inner.write().updated = false;
    }

    fn snapshot(&self) -> ChannelSnapshot {
        let g = self.inner.read();
        ChannelSnapshot { kind: self.kind.clone(), value: g.value.clone(), updated: g.updated }
    }

    fn restore(&self, snap: ChannelSnapshot) -> GraphResult<()> {
        let mut g = self.inner.write();
        g.value = snap.value;
        g.updated = snap.updated;
        Ok(())
    }

    fn begin_step(&self) {
        if matches!(self.kind, ChannelKind::Ephemeral) {
            let mut g = self.inner.write();
            g.value = None;
        }
    }
}

fn reduce_binary(op: &BinaryOp, acc: Option<Value>, next: Value) -> GraphResult<Value> {
    match (op, acc, next) {
        (BinaryOp::Add, None, n) => Ok(n),
        (BinaryOp::Add, Some(a), n) => add_values(a, n),
        (BinaryOp::Extend, None, n) => Ok(extend_arrayish(Value::Array(vec![]), n)),
        (BinaryOp::Extend, Some(a), n) => Ok(extend_arrayish(a, n)),
        (BinaryOp::MergeDicts, None, n) => Ok(merge_dicts(Value::Object(Default::default()), n)),
        (BinaryOp::MergeDicts, Some(a), n) => Ok(merge_dicts(a, n)),
    }
}

fn add_values(a: Value, b: Value) -> GraphResult<Value> {
    use serde_json::Number;
    match (&a, &b) {
        (Value::Number(x), Value::Number(y)) => {
            if let (Some(xi), Some(yi)) = (x.as_i64(), y.as_i64()) {
                return Ok(Value::Number(Number::from(xi + yi)));
            }
            let xf = x.as_f64().unwrap_or(0.0);
            let yf = y.as_f64().unwrap_or(0.0);
            Number::from_f64(xf + yf).map(Value::Number).ok_or_else(
                || GraphError::InvalidUpdate { channel: "<binop>".into(), reason: "non-finite sum".into() },
            )
        }
        (Value::String(x), Value::String(y)) => Ok(Value::String(format!("{x}{y}"))),
        (Value::Array(x), Value::Array(y)) => {
            let mut out = x.clone();
            out.extend(y.clone());
            Ok(Value::Array(out))
        }
        _ => Err(GraphError::InvalidUpdate {
            channel: "<binop:add>".into(),
            reason: format!("cannot add {a:?} + {b:?}"),
        }),
    }
}

fn extend_arrayish(acc: Value, next: Value) -> Value {
    let mut out = match acc {
        Value::Array(a) => a,
        Value::Null => Vec::new(),
        other => vec![other],
    };
    match next {
        Value::Array(a) => out.extend(a),
        other => out.push(other),
    }
    Value::Array(out)
}

fn merge_dicts(acc: Value, next: Value) -> Value {
    let mut out = match acc {
        Value::Object(o) => o,
        _ => Default::default(),
    };
    if let Value::Object(o) = next {
        for (k, v) in o {
            out.insert(k, v);
        }
    }
    Value::Object(out)
}

/// LangChain-style `add_messages` reducer:
///  - if message has same `id` as existing, replace it
///  - if message has `type == "remove"`, drop the matching id
///  - otherwise append
fn add_messages_into(left: &mut Vec<Value>, update: Value) {
    let push_one = |left: &mut Vec<Value>, msg: Value| {
        if let Some(id) = msg.get("id").and_then(|v| v.as_str()).map(String::from) {
            if let Some(pos) = left.iter().position(|m| m.get("id").and_then(|v| v.as_str()) == Some(&id)) {
                if msg.get("type").and_then(|v| v.as_str()) == Some("remove") {
                    left.remove(pos);
                } else {
                    left[pos] = msg;
                }
                return;
            }
        }
        // message has no id or wasn't found
        if msg.get("type").and_then(|v| v.as_str()) == Some("remove") {
            return;
        }
        left.push(msg);
    };
    match update {
        Value::Array(arr) => {
            for m in arr {
                push_one(left, m);
            }
        }
        Value::Object(_) => push_one(left, update),
        Value::String(s) => left.push(json!({ "type": "human", "content": s })),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_value_overwrites() {
        let c = StoredChannel::new(ChannelKind::LastValue);
        c.apply(vec![json!(1), json!(2), json!(3)]).unwrap();
        assert_eq!(c.get(), Some(json!(3)));
    }

    #[test]
    fn topic_accumulates() {
        let c = StoredChannel::new(ChannelKind::Topic { unique: false, accumulate: true });
        c.apply(vec![json!("a"), json!("b")]).unwrap();
        c.apply(vec![json!("c")]).unwrap();
        assert_eq!(c.get(), Some(json!(["a", "b", "c"])));
    }

    #[test]
    fn topic_unique() {
        let c = StoredChannel::new(ChannelKind::Topic { unique: true, accumulate: true });
        c.apply(vec![json!("a"), json!("b"), json!("a")]).unwrap();
        assert_eq!(c.get(), Some(json!(["a", "b"])));
    }

    #[test]
    fn binary_add_numbers() {
        let c = StoredChannel::new(ChannelKind::BinaryOperator { op: BinaryOp::Add });
        c.apply(vec![json!(1), json!(2), json!(3)]).unwrap();
        assert_eq!(c.get(), Some(json!(6)));
    }

    #[test]
    fn binary_extend_arrays() {
        let c = StoredChannel::new(ChannelKind::BinaryOperator { op: BinaryOp::Extend });
        c.apply(vec![json!(["a"]), json!(["b", "c"])]).unwrap();
        assert_eq!(c.get(), Some(json!(["a", "b", "c"])));
    }

    #[test]
    fn merge_dicts() {
        let c = StoredChannel::new(ChannelKind::BinaryOperator { op: BinaryOp::MergeDicts });
        c.apply(vec![json!({"a":1}), json!({"b":2,"a":3})]).unwrap();
        assert_eq!(c.get(), Some(json!({"a":3, "b":2})));
    }

    #[test]
    fn ephemeral_clears_on_step() {
        let c = StoredChannel::new(ChannelKind::Ephemeral);
        c.apply(vec![json!("once")]).unwrap();
        assert_eq!(c.get(), Some(json!("once")));
        c.begin_step();
        assert_eq!(c.get(), None);
    }

    #[test]
    fn add_messages_appends_and_replaces() {
        let c = StoredChannel::new(ChannelKind::AddMessages);
        c.apply(vec![json!({"id":"1", "content":"hi"})]).unwrap();
        c.apply(vec![json!({"id":"2", "content":"yo"})]).unwrap();
        c.apply(vec![json!({"id":"1", "content":"hi!"})]).unwrap();
        let v = c.get().unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["content"], "hi!");
    }

    #[test]
    fn add_messages_remove() {
        let c = StoredChannel::new(ChannelKind::AddMessages);
        c.apply(vec![json!({"id":"1", "content":"hi"})]).unwrap();
        c.apply(vec![json!({"id":"1", "type":"remove"})]).unwrap();
        assert_eq!(c.get(), Some(json!([])));
    }

    #[test]
    fn snapshot_roundtrip() {
        let c = StoredChannel::new(ChannelKind::LastValue);
        c.apply(vec![json!({"x": 1})]).unwrap();
        let s = c.snapshot();
        let c2 = StoredChannel::from_snapshot(s);
        assert_eq!(c2.get(), Some(json!({"x": 1})));
    }
}
