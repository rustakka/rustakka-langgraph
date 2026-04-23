//! `create_swarm` — peer-to-peer multi-agent handoff pattern.
//!
//! Mirrors `langgraph-swarm.create_swarm(...)`. Each agent can "hand off"
//! to any other agent by emitting `{"next": "<agent_name>"}` in its
//! returned state. When no handoff is requested the graph finishes.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::Value;

use rustakka_langgraph_core::errors::{GraphError, GraphResult};
use rustakka_langgraph_core::graph::{CompileConfig, CompiledStateGraph, StateGraph, END, START};
use rustakka_langgraph_core::state::{ChannelSpec, DynamicState};

use crate::supervisor::Agent;

/// Build a swarm graph where every agent is reachable from every other
/// agent via the `next` channel. `default_agent` is entered first.
///
/// Agents signal a handoff by writing `"next": "<name>"` into state. If
/// the value is missing (or is `"__end__"`) the graph finishes.
pub async fn create_swarm(
    agents: Vec<Agent>,
    default_agent: impl Into<String>,
) -> GraphResult<CompiledStateGraph> {
    if agents.is_empty() {
        return Err(GraphError::Compile(
            "create_swarm requires at least one agent".into(),
        ));
    }
    let default_agent = default_agent.into();
    let mut g = StateGraph::<DynamicState>::new();
    g.add_channel(ChannelSpec::add_messages("messages"));
    g.add_channel(ChannelSpec::last_value("next"));

    let names: Vec<String> = agents.iter().map(|a| a.name.clone()).collect();
    if !names.iter().any(|n| n == &default_agent) {
        return Err(GraphError::Compile(format!(
            "default_agent `{default_agent}` is not in the agent list",
        )));
    }
    for a in agents {
        g.add_node(a.name.clone(), a.node)?;
    }
    g.add_edge(START, default_agent.clone());

    // Every agent may route to every other agent (and END) via the `next` channel.
    let mut branches: HashMap<String, String> = HashMap::new();
    for n in &names {
        branches.insert(n.clone(), n.clone());
    }
    branches.insert("END".into(), END.into());
    branches.insert(END.into(), END.into());
    let router: Arc<
        dyn Fn(&BTreeMap<String, Value>) -> Vec<String> + Send + Sync + 'static,
    > = Arc::new(|values: &BTreeMap<String, Value>| {
        match values.get("next").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => vec![n.to_string()],
            _ => vec![END.into()],
        }
    });
    for n in &names {
        g.add_conditional_edges(n.clone(), router.clone(), Some(branches.clone()));
    }
    g.compile(CompileConfig::default()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustakka_langgraph_core::config::RunnableConfig;
    use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
    use rustakka_langgraph_core::runner::invoke_dynamic;
    use serde_json::json;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn agent_can_hand_off_to_another_agent() {
        let agent_a = Agent::new(
            "a",
            NodeKind::from_fn(|_| async move {
                let mut m = BTreeMap::new();
                m.insert("messages".into(), json!([{"content": "hi from a"}]));
                m.insert("next".into(), json!("b"));
                Ok(NodeOutput::Update(m))
            }),
        );
        let agent_b = Agent::new(
            "b",
            NodeKind::from_fn(|_| async move {
                let mut m = BTreeMap::new();
                m.insert("messages".into(), json!([{"content": "hi from b"}]));
                m.insert("next".into(), json!(""));
                Ok(NodeOutput::Update(m))
            }),
        );
        let graph = create_swarm(vec![agent_a, agent_b], "a").await.unwrap();
        let values = invoke_dynamic(
            Arc::new(graph),
            BTreeMap::new(),
            RunnableConfig::default(),
        )
        .await
        .unwrap();
        let contents: Vec<&str> = values["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .collect();
        assert!(contents.contains(&"hi from a"));
        assert!(contents.contains(&"hi from b"));
    }
}
