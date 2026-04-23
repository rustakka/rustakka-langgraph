//! `create_supervisor` — classic hub-and-spoke multi-agent pattern.
//!
//! Mirrors `langgraph-supervisor.create_supervisor(...)`. A *supervisor*
//! node routes to one of `agents` based on the supervisor's own output, and
//! every agent returns to the supervisor using
//! `Command(graph=PARENT, goto="supervisor")` — implemented here by a
//! simple "back-edge" from each agent to the supervisor node.
//!
//! Termination: when the supervisor returns an empty goto (or selects the
//! sentinel `__end__`) the graph finishes.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::Value;

use rustakka_langgraph_core::errors::GraphResult;
use rustakka_langgraph_core::graph::{CompileConfig, CompiledStateGraph, StateGraph, END, START};
use rustakka_langgraph_core::node::NodeKind;
use rustakka_langgraph_core::state::{ChannelSpec, DynamicState};

/// A named sub-agent. `node` must be a fully-formed node (typically another
/// compiled graph wrapped via `CompiledStateGraph::as_subgraph_invoker`).
pub struct Agent {
    pub name: String,
    pub node: NodeKind,
}

impl Agent {
    pub fn new(name: impl Into<String>, node: NodeKind) -> Self {
        Self { name: name.into(), node }
    }
}

/// Router function: inspects current state and returns the next agent name
/// or [`END`]. This is usually driven by an LLM that sees the full message
/// history.
pub type SupervisorRouter = Arc<
    dyn Fn(&BTreeMap<String, Value>) -> Vec<String> + Send + Sync + 'static,
>;

/// Build a supervisor graph:
///
/// ```text
///     START → supervisor → (agent_1 | agent_2 | … | END)
///     agent_i → supervisor   (looping)
/// ```
///
/// `supervisor_node` is the node that implements the supervisor's own
/// logic (typically a chat model). `router` inspects state *after* the
/// supervisor runs and returns either `"END"` / [`END`] or one of the agent
/// names registered in `agents`.
pub async fn create_supervisor(
    supervisor_node: NodeKind,
    router: SupervisorRouter,
    agents: Vec<Agent>,
) -> GraphResult<CompiledStateGraph> {
    if agents.is_empty() {
        return Err(rustakka_langgraph_core::errors::GraphError::Compile(
            "create_supervisor requires at least one agent".into(),
        ));
    }
    let mut g = StateGraph::<DynamicState>::new();
    g.add_channel(ChannelSpec::add_messages("messages"));

    const SUPERVISOR: &str = "supervisor";
    g.add_node(SUPERVISOR, supervisor_node)?;
    for a in &agents {
        g.add_node(a.name.clone(), clone_node(&a.node))?;
    }
    g.add_edge(START, SUPERVISOR);

    // Supervisor routes dynamically to one of the agents (or END).
    let mut branches: HashMap<String, String> = HashMap::new();
    for a in &agents {
        branches.insert(a.name.clone(), a.name.clone());
    }
    branches.insert("END".into(), END.into());
    branches.insert(END.into(), END.into());
    g.add_conditional_edges(SUPERVISOR, router, Some(branches));

    // Every agent loops back to the supervisor.
    for a in &agents {
        g.add_edge(a.name.clone(), SUPERVISOR);
    }

    g.compile(CompileConfig::default()).await
}

fn clone_node(n: &NodeKind) -> NodeKind {
    match n {
        NodeKind::Rust(f) => NodeKind::Rust(f.clone()),
        NodeKind::Python(p) => NodeKind::Python(p.clone()),
        NodeKind::Subgraph(s) => NodeKind::Subgraph(s.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustakka_langgraph_core::config::RunnableConfig;
    use rustakka_langgraph_core::node::NodeOutput;
    use rustakka_langgraph_core::runner::invoke_dynamic;
    use serde_json::json;

    fn constant_node(msg: &'static str) -> NodeKind {
        NodeKind::from_fn(move |_| async move {
            let mut m = BTreeMap::new();
            m.insert(
                "messages".into(),
                json!([{ "role": "assistant", "content": msg }]),
            );
            Ok(NodeOutput::Update(m))
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn supervisor_routes_to_agent_then_finishes() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = counter.clone();
        let supervisor = NodeKind::from_fn(move |_| {
            let c = c.clone();
            async move {
                let n = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let tag = if n == 0 { "call" } else { "done" };
                let mut m = BTreeMap::new();
                m.insert(
                    "messages".into(),
                    json!([{ "role": "assistant", "content": tag }]),
                );
                Ok(NodeOutput::Update(m))
            }
        });
        let router: SupervisorRouter = Arc::new(|values: &BTreeMap<String, Value>| {
            let last = values
                .get("messages")
                .and_then(|v| v.as_array())
                .and_then(|a| a.last())
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");
            match last {
                "call" => vec!["worker".into()],
                _ => vec![END.into()],
            }
        });
        let graph = create_supervisor(
            supervisor,
            router,
            vec![Agent::new("worker", constant_node("worked"))],
        )
        .await
        .unwrap();
        let values = invoke_dynamic(
            Arc::new(graph),
            BTreeMap::new(),
            RunnableConfig::default(),
        )
        .await
        .unwrap();
        // Expect at least one supervisor and one worker turn.
        let msgs = values["messages"].as_array().unwrap().clone();
        let contents: Vec<&str> = msgs
            .iter()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .collect();
        assert!(contents.contains(&"call"));
        assert!(contents.contains(&"worked"));
        assert!(contents.contains(&"done"));
    }
}
