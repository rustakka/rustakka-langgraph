//! `create_react_agent` — assembles a typical ReAct loop:
//! `(agent → tools)` cycling until the LLM returns no `tool_calls`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

use rustakka_langgraph_core::errors::GraphResult;
use rustakka_langgraph_core::graph::{CompileConfig, CompiledStateGraph, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::state::{ChannelSpec, DynamicState};

use crate::tool_node::{tools_condition, Tool, ToolNode};

#[derive(Debug, Clone, Default)]
pub struct ReactAgentOptions {
    pub system_prompt: Option<String>,
    pub recursion_limit: Option<u32>,
}

/// `model_fn(messages, system_prompt) -> message`.
pub type ModelFn = Arc<
    dyn Fn(Vec<Value>, Option<String>) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = GraphResult<Value>> + Send>,
        > + Send
        + Sync,
>;

pub async fn create_react_agent(
    model: ModelFn,
    tools: Vec<Tool>,
    opts: ReactAgentOptions,
) -> GraphResult<CompiledStateGraph> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_channel(ChannelSpec::add_messages("messages"));

    let model_clone = model.clone();
    let sys_prompt = opts.system_prompt.clone();
    g.add_node(
        "agent",
        NodeKind::from_fn(move |input: BTreeMap<String, Value>| {
            let model = model_clone.clone();
            let sys = sys_prompt.clone();
            async move {
                let msgs = input
                    .get("messages")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                let new_msg = (model)(msgs, sys).await?;
                let mut out = BTreeMap::new();
                out.insert("messages".into(), Value::Array(vec![new_msg]));
                Ok(NodeOutput::Update(out))
            }
        }),
    )?;
    g.add_node("tools", ToolNode::new(tools).into_node())?;

    g.add_edge(START, "agent");
    g.add_conditional_edges("agent", Arc::new(tools_condition), None);
    g.add_edge("tools", "agent");
    g.add_edge("agent", END);

    let mut cfg = CompileConfig::default();
    cfg.recursion_limit = opts.recursion_limit;
    g.compile(cfg).await
}
