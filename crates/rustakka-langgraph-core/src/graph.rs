//! `StateGraph` builder + `CompiledStateGraph` runtime handle.
//!
//! Mirrors `langgraph.graph.StateGraph` and `langgraph.graph.CompiledStateGraph`
//! at the public API level. Compilation snapshots the topology into an
//! immutable [`GraphTopology`] handed to a [`crate::coordinator::GraphCoordinator`]
//! at runtime.

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::command::Command;
use crate::config::{RunnableConfig, StreamMode};
use crate::errors::{GraphError, GraphResult};
use crate::node::{NodeKind, NodeOutput, SubgraphInvoker};
use crate::state::{ChannelSpec, GraphState, GraphValues};
use crate::stream::{StreamBus, StreamEvent};

/// Sentinel "start" node name (matches upstream `langgraph.constants.START`).
pub const START: &str = "__start__";
/// Sentinel "end" node name (matches upstream `langgraph.constants.END`).
pub const END: &str = "__end__";

// ---------------------------- builder side ----------------------------

/// Optional retry policy attached to nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms: u64,
}

/// Function used to derive a cache key from a node's input map. Mirrors
/// upstream's `CachePolicy(key_func=...)`.
pub type CacheKeyFn = Arc<
    dyn Fn(&BTreeMap<String, Value>) -> String + Send + Sync + 'static,
>;

/// Per-node cache policy. If a node produces the same `key_func(input)`
/// within `ttl_seconds` (unbounded when `None`), the prior `NodeOutput` is
/// replayed instead of re-executing.
#[derive(Clone)]
pub struct CachePolicy {
    pub key_func: CacheKeyFn,
    pub ttl_seconds: Option<u64>,
}

impl std::fmt::Debug for CachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachePolicy")
            .field("ttl_seconds", &self.ttl_seconds)
            .finish()
    }
}

impl CachePolicy {
    /// Default: serialize the whole input map as JSON for the cache key.
    pub fn default_key() -> CacheKeyFn {
        Arc::new(|input: &BTreeMap<String, Value>| {
            serde_json::to_string(input).unwrap_or_default()
        })
    }

    pub fn new(ttl_seconds: Option<u64>) -> Self {
        Self { key_func: Self::default_key(), ttl_seconds }
    }

    pub fn with_key(mut self, f: CacheKeyFn) -> Self {
        self.key_func = f;
        self
    }
}

/// Conditional edge: a router function evaluated on the current state map
/// returning the next node target(s).
pub type RouterFn = Arc<
    dyn Fn(&BTreeMap<String, Value>) -> Vec<String> + Send + Sync + 'static,
>;

#[derive(Clone)]
pub struct ConditionalEdge {
    pub source: String,
    pub router: RouterFn,
    /// Optional explicit branch->target mapping (mirrors upstream's
    /// `path_map` argument).
    pub branches: Option<HashMap<String, String>>,
}

impl std::fmt::Debug for ConditionalEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalEdge")
            .field("source", &self.source)
            .field("branches", &self.branches)
            .finish()
    }
}

#[derive(Default)]
struct GraphBuild {
    nodes: HashMap<String, NodeKind>,
    /// (source, target) — `source == START` means input edges.
    static_edges: Vec<(String, String)>,
    conditional_edges: Vec<ConditionalEdge>,
    entry_point: Option<String>,
    finish_point: Option<String>,
    extra_channels: Vec<ChannelSpec>,
    retries: HashMap<String, RetryPolicy>,
    caches: HashMap<String, CachePolicy>,
}

/// Builder counterpart of `langgraph.graph.StateGraph(state_schema)`.
pub struct StateGraph<S: GraphState> {
    build: GraphBuild,
    _state: PhantomData<S>,
}

impl<S: GraphState> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GraphState> StateGraph<S> {
    pub fn new() -> Self {
        Self { build: GraphBuild::default(), _state: PhantomData }
    }

    pub fn add_node(&mut self, name: impl Into<String>, node: NodeKind) -> GraphResult<&mut Self> {
        let name = name.into();
        if name == START || name == END {
            return Err(GraphError::Compile(format!("`{name}` is reserved")));
        }
        if self.build.nodes.contains_key(&name) {
            return Err(GraphError::DuplicateNode(name));
        }
        self.build.nodes.insert(name, node);
        Ok(self)
    }

    pub fn add_edge(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> &mut Self {
        self.build.static_edges.push((source.into(), target.into()));
        self
    }

    pub fn add_conditional_edges(
        &mut self,
        source: impl Into<String>,
        router: RouterFn,
        branches: Option<HashMap<String, String>>,
    ) -> &mut Self {
        self.build.conditional_edges.push(ConditionalEdge {
            source: source.into(),
            router,
            branches,
        });
        self
    }

    pub fn set_entry_point(&mut self, name: impl Into<String>) -> &mut Self {
        self.build.entry_point = Some(name.into());
        self
    }

    pub fn set_finish_point(&mut self, name: impl Into<String>) -> &mut Self {
        self.build.finish_point = Some(name.into());
        self
    }

    /// Manually declare additional channels not covered by the schema.
    pub fn add_channel(&mut self, spec: ChannelSpec) -> &mut Self {
        self.build.extra_channels.push(spec);
        self
    }

    pub fn set_retry_policy(&mut self, name: impl Into<String>, policy: RetryPolicy) -> &mut Self {
        self.build.retries.insert(name.into(), policy);
        self
    }

    /// Attach a [`CachePolicy`] to `node`. Subsequent invocations with the
    /// same `key_func(input)` replay the cached [`NodeOutput`] (subject to
    /// `ttl_seconds`). Mirrors upstream
    /// `add_node(..., cache_policy=CachePolicy(...))`.
    pub fn set_cache_policy(
        &mut self,
        name: impl Into<String>,
        policy: CachePolicy,
    ) -> &mut Self {
        self.build.caches.insert(name.into(), policy);
        self
    }

    /// Validate + freeze into a [`CompiledStateGraph`].
    pub async fn compile(self, cfg: CompileConfig) -> GraphResult<CompiledStateGraph> {
        let GraphBuild {
            nodes,
            static_edges,
            conditional_edges,
            entry_point,
            finish_point,
            extra_channels,
            retries,
            caches,
        } = self.build;

        if nodes.is_empty() {
            return Err(GraphError::Compile("graph has no nodes".into()));
        }
        for (s, t) in &static_edges {
            if s != START && !nodes.contains_key(s) {
                return Err(GraphError::UnknownNode(s.clone()));
            }
            if t != END && !nodes.contains_key(t) {
                return Err(GraphError::UnknownNode(t.clone()));
            }
        }
        for c in &conditional_edges {
            if c.source != START && !nodes.contains_key(&c.source) {
                return Err(GraphError::UnknownNode(c.source.clone()));
            }
        }
        let entry_targets: Vec<String> = static_edges
            .iter()
            .filter_map(|(s, t)| if s == START { Some(t.clone()) } else { None })
            .collect();
        let entry_targets = if entry_targets.is_empty() {
            entry_point
                .as_ref()
                .map(|e| vec![e.clone()])
                .ok_or(GraphError::MissingEntryPoint)?
        } else {
            entry_targets
        };

        let mut specs = S::channel_specs();
        specs.extend(extra_channels);

        let topology = Arc::new(GraphTopology {
            nodes,
            static_edges,
            conditional_edges,
            entry_targets,
            finish_point,
            channel_specs: specs,
            retries,
            caches,
            cfg,
        });

        Ok(CompiledStateGraph {
            topology,
            cache: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompileConfig {
    pub recursion_limit: Option<u32>,
    pub debug: bool,
    /// If set, the compiled graph attaches this checkpointer factory id at
    /// compile time. The actual saver is supplied at runtime via
    /// `CompiledStateGraph::with_checkpointer`.
    pub checkpointer_id: Option<String>,
    /// Static breakpoint: pause with an Interrupt *before* any of these nodes
    /// executes. Mirrors upstream `compile(interrupt_before=[...])`.
    #[serde(default)]
    pub interrupt_before: Vec<String>,
    /// Static breakpoint: pause with an Interrupt *after* any of these nodes
    /// writes. Mirrors upstream `compile(interrupt_after=[...])`.
    #[serde(default)]
    pub interrupt_after: Vec<String>,
    /// Checkpoint-flush policy: `Sync` persists after every update phase
    /// (default, strongest guarantee), `Async` spawns the saver call,
    /// `Exit` defers all writes until the run completes. Mirrors upstream
    /// `compile(durability=...)`.
    #[serde(default)]
    pub durability: Durability,
}

/// Checkpoint-flush timing. Mirrors upstream's `durability` argument
/// (`"sync" | "async" | "exit"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    /// Await the saver inside the update phase (safest, default).
    #[default]
    Sync,
    /// Spawn the saver call and continue; errors are surfaced via tracing.
    Async,
    /// Only persist at run-exit (final step).
    Exit,
}

// ---------------------------- compiled side ----------------------------

/// Immutable, thread-safe runtime topology shared by every invocation.
pub struct GraphTopology {
    pub nodes: HashMap<String, NodeKind>,
    pub static_edges: Vec<(String, String)>,
    pub conditional_edges: Vec<ConditionalEdge>,
    pub entry_targets: Vec<String>,
    pub finish_point: Option<String>,
    pub channel_specs: Vec<ChannelSpec>,
    pub retries: HashMap<String, RetryPolicy>,
    pub caches: HashMap<String, CachePolicy>,
    pub cfg: CompileConfig,
}

/// Entry in the per-graph node-output cache.
#[derive(Clone)]
pub struct CacheEntry {
    pub value: BTreeMap<String, Value>,
    pub expires_at: Option<std::time::Instant>,
}

pub type NodeCache = parking_lot::RwLock<HashMap<(String, String), CacheEntry>>;

impl std::fmt::Debug for GraphTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphTopology")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("static_edges", &self.static_edges)
            .field("entry_targets", &self.entry_targets)
            .finish()
    }
}

impl GraphTopology {
    /// Compute next-step targets after a node finishes, given its emitted
    /// command (if any) and the post-update state values.
    pub fn next_targets(
        &self,
        node: &str,
        cmd: Option<&Command>,
        values: &BTreeMap<String, Value>,
    ) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(c) = cmd {
            out.extend(c.goto.iter().cloned());
        }
        for (s, t) in &self.static_edges {
            if s == node {
                out.push(t.clone());
            }
        }
        for ce in &self.conditional_edges {
            if ce.source == node {
                let raw = (ce.router)(values);
                if let Some(map) = &ce.branches {
                    for r in raw {
                        if let Some(target) = map.get(&r) {
                            out.push(target.clone());
                        } else if self.nodes.contains_key(&r) || r == END {
                            out.push(r);
                        }
                    }
                } else {
                    out.extend(raw);
                }
            }
        }
        // dedupe while preserving order
        let mut seen = std::collections::HashSet::new();
        out.retain(|t| seen.insert(t.clone()));
        out
    }
}

/// Compiled graph handle returned by `StateGraph::compile` — the runtime
/// counterpart of `langgraph.graph.CompiledStateGraph`.
#[derive(Clone)]
pub struct CompiledStateGraph {
    pub(crate) topology: Arc<GraphTopology>,
    /// Per-graph node-output cache keyed by `(node_name, cache_key)`.
    /// Shared across invocations of the same compiled graph.
    pub(crate) cache: Arc<NodeCache>,
}

impl CompiledStateGraph {
    pub fn topology(&self) -> &Arc<GraphTopology> {
        &self.topology
    }

    pub fn node_cache(&self) -> &Arc<NodeCache> {
        &self.cache
    }

    pub fn channel_specs(&self) -> &[ChannelSpec] {
        &self.topology.channel_specs
    }

    /// Expose this compiled graph as a `SubgraphInvoker` so it can be plugged
    /// in as a node inside a parent graph.
    pub fn as_subgraph_invoker(self: Arc<Self>) -> Arc<dyn SubgraphInvoker> {
        Arc::new(SubgraphAdapter { inner: self })
    }

    /// Render this graph as a Mermaid `flowchart TD` diagram. Mirrors
    /// upstream `get_graph().draw_mermaid()`.
    pub fn draw_mermaid(&self) -> String {
        crate::visualize::draw_mermaid(self)
    }

    /// Render a minimal ASCII overview (nodes + edges). Mirrors upstream
    /// `get_graph().draw_ascii()`.
    pub fn draw_ascii(&self) -> String {
        crate::visualize::draw_ascii(self)
    }
}

struct SubgraphAdapter {
    inner: Arc<CompiledStateGraph>,
}

impl SubgraphInvoker for SubgraphAdapter {
    fn invoke(
        &self,
        input: BTreeMap<String, Value>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = GraphResult<NodeOutput>> + Send + 'static>>
    {
        let inner = self.inner.clone();
        Box::pin(async move {
            let cfg = RunnableConfig::default();

            // If the parent coordinator installed a `CURRENT_WRITER` for the
            // subgraph node, inherit its bus so events from the child graph
            // surface to parent subscribers with a namespaced prefix.
            let parent_ctx = crate::stream::current_writer();

            let res = crate::runner::run_internal_with_parent(
                inner,
                input,
                cfg,
                None,
                None,
                parent_ctx.map(|w| {
                    let mut ns = w.namespace().to_vec();
                    ns.push(w.node().to_string());
                    (w.bus().clone(), ns)
                }),
            )
            .await?;
            if let Some(cmd) = res.parent_command {
                return Ok(NodeOutput::Command(cmd));
            }
            Ok(NodeOutput::Update(res.values))
        })
    }
}

// silence unused import warnings (mpsc / GraphValues are used by runner via re-export)
const _: fn() = || {
    let _: Option<mpsc::UnboundedReceiver<StreamEvent>> = None;
    let _: Option<GraphValues> = None;
    let _: Option<StreamMode> = None;
    let _: Option<Arc<StreamBus>> = None;
};
