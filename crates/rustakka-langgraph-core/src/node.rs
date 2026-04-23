//! `NodeKind` enum + node invocation glue.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::command::{Command, Interrupt};
use crate::errors::GraphResult;

/// What a node returns. Either a plain "channel writes" map (most common),
/// a [`Command`] for richer control flow, or an [`Interrupt`] payload.
#[derive(Debug, Clone)]
pub enum NodeOutput {
    /// Map of channel-name -> update value (the Pythonic dict return).
    Update(BTreeMap<String, Value>),
    /// Full Command response (`goto`/`update`/`send`/`resume`).
    Command(Command),
    /// Node hit `interrupt(value)`; coordinator must persist + surface.
    Interrupted(Interrupt),
    /// Equivalent to "vote to halt": no updates, no goto.
    Halt,
}

impl NodeOutput {
    pub fn empty() -> Self {
        NodeOutput::Halt
    }
    pub fn from_value(v: Value) -> Self {
        match v {
            Value::Object(map) => {
                let mut out = BTreeMap::new();
                for (k, vv) in map {
                    out.insert(k, vv);
                }
                NodeOutput::Update(out)
            }
            Value::Null => NodeOutput::Halt,
            other => {
                let mut out = BTreeMap::new();
                out.insert("output".into(), other);
                NodeOutput::Update(out)
            }
        }
    }
}

/// Boxed async signature for a Rust node.
pub type BoxNodeFuture = Pin<Box<dyn Future<Output = GraphResult<NodeOutput>> + Send + 'static>>;

/// Native Rust node closure.
pub type NodeFn = Arc<dyn Fn(BTreeMap<String, Value>) -> BoxNodeFuture + Send + Sync>;

/// Variants of node implementations the engine knows about. The Python
/// variant is supplied by the `pylanggraph` PyO3 crate via [`PyCallableNode`].
pub enum NodeKind {
    /// Pure-Rust async closure operating on the materialized values map.
    Rust(NodeFn),
    /// Python-callable node — opaque handle resolved by the bindings layer.
    Python(Arc<dyn PyCallableNode>),
    /// Compiled subgraph: the coordinator dispatches the input state to a
    /// nested compiled graph and treats its output as channel writes.
    Subgraph(Arc<dyn SubgraphInvoker>),
}

impl Clone for NodeKind {
    fn clone(&self) -> Self {
        self.clone_kind()
    }
}

impl std::fmt::Debug for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeKind::Rust(_) => f.write_str("NodeKind::Rust(<closure>)"),
            NodeKind::Python(_) => f.write_str("NodeKind::Python(<py>)"),
            NodeKind::Subgraph(_) => f.write_str("NodeKind::Subgraph(<compiled>)"),
        }
    }
}

/// Trait the PyO3 layer implements to forward a call to a Python callable.
pub trait PyCallableNode: Send + Sync + 'static {
    fn call(
        &self,
        input: BTreeMap<String, Value>,
    ) -> Pin<Box<dyn Future<Output = GraphResult<NodeOutput>> + Send + 'static>>;
}

/// Subgraph invocation hook (avoids a generic parameter on `NodeKind`).
pub trait SubgraphInvoker: Send + Sync + 'static {
    fn invoke(
        &self,
        input: BTreeMap<String, Value>,
    ) -> Pin<Box<dyn Future<Output = GraphResult<NodeOutput>> + Send + 'static>>;
}

impl NodeKind {
    pub fn from_fn<F, Fut>(f: F) -> Self
    where
        F: Fn(BTreeMap<String, Value>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = GraphResult<NodeOutput>> + Send + 'static,
    {
        let arc: NodeFn = Arc::new(move |i| Box::pin(f(i)) as BoxNodeFuture);
        NodeKind::Rust(arc)
    }

    pub async fn invoke(&self, input: BTreeMap<String, Value>) -> GraphResult<NodeOutput> {
        match self {
            NodeKind::Rust(f) => f(input).await,
            NodeKind::Python(p) => p.call(input).await,
            NodeKind::Subgraph(s) => s.invoke(input).await,
        }
    }

    /// Shallow-clone the underlying handle (Arc bump). `NodeKind` doesn't
    /// derive `Clone` because the variants hold boxed trait objects; this
    /// is the cheap structural copy used by the coordinator and node
    /// workers when dispatching.
    pub fn clone_kind(&self) -> Self {
        match self {
            NodeKind::Rust(f) => NodeKind::Rust(f.clone()),
            NodeKind::Python(p) => NodeKind::Python(p.clone()),
            NodeKind::Subgraph(s) => NodeKind::Subgraph(s.clone()),
        }
    }
}
