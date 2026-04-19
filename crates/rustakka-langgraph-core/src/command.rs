//! Control-flow primitives mirroring `langgraph.types.{Command, Send, Interrupt}`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Routing target inside a graph: either a node name, [`END`], or START.
pub type Target = String;

/// Mirror of `langgraph.types.Send(node, arg)`. Use this from a node return
/// to dispatch additional `(node, state)` invocations within the current
/// superstep (fan-out / map-reduce).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Send {
    pub node: String,
    pub arg: Value,
}

impl Send {
    pub fn new(node: impl Into<String>, arg: impl Into<Value>) -> Self {
        Self { node: node.into(), arg: arg.into() }
    }
}

/// Mirror of `langgraph.types.Command(...)`.
///
/// Returning a `Command` from a node lets that node simultaneously:
///   - update channels (`update`)
///   - choose the next node(s) to run (`goto`)
///   - resume from an `interrupt(...)` (`resume`)
///   - target a parent graph from inside a subgraph (`graph`)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Command {
    /// Channel writes to apply during the Update phase.
    #[serde(default)]
    pub update: BTreeMap<String, Value>,
    /// Next nodes (or [`END`]) to dispatch in the next superstep.
    #[serde(default)]
    pub goto: Vec<Target>,
    /// Fan-out sends to dispatch in the next superstep.
    #[serde(default)]
    pub send: Vec<Send>,
    /// Resume value for a previously-interrupted execution.
    #[serde(default)]
    pub resume: Option<Value>,
    /// Target parent graph (`Some("PARENT")`) when used inside a subgraph.
    #[serde(default)]
    pub graph: Option<String>,
}

impl Command {
    pub fn goto(target: impl Into<String>) -> Self {
        Self { goto: vec![target.into()], ..Default::default() }
    }
    pub fn update(map: BTreeMap<String, Value>) -> Self {
        Self { update: map, ..Default::default() }
    }
    pub fn with_update(mut self, k: impl Into<String>, v: impl Into<Value>) -> Self {
        self.update.insert(k.into(), v.into());
        self
    }
    pub fn with_goto(mut self, t: impl Into<String>) -> Self {
        self.goto.push(t.into());
        self
    }
    pub fn with_send(mut self, s: Send) -> Self {
        self.send.push(s);
        self
    }
    pub fn resume(v: impl Into<Value>) -> Self {
        Self { resume: Some(v.into()), ..Default::default() }
    }
}

/// Raised by a node via `interrupt(value)`. Surfaces as
/// `GraphError::NodeInterrupt` to the caller; resume by re-invoking with
/// `Command::resume(...)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    pub node: String,
    pub value: Value,
    /// Stable identifier for resumability across runs.
    pub id: String,
}

impl Interrupt {
    pub fn new(node: impl Into<String>, value: impl Into<Value>) -> Self {
        Self { node: node.into(), value: value.into(), id: uuid::Uuid::new_v4().to_string() }
    }
}
