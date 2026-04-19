//! Pure-Rust ping/pong graph: two nodes bouncing a counter until it hits 5.
use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::invoke_dynamic;
use rustakka_langgraph_core::state::{ChannelSpec, DynamicState};
use serde_json::json;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_channel(ChannelSpec::last_value("count"));

    let bump = |delta: i64| {
        NodeKind::from_fn(move |input: BTreeMap<String, serde_json::Value>| async move {
            let cur = input.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut m = BTreeMap::new();
            m.insert("count".into(), json!(cur + delta));
            Ok(NodeOutput::Update(m))
        })
    };
    g.add_node("ping", bump(1)).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.add_node("pong", bump(1)).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.add_edge(START, "ping");
    g.add_edge("ping", "pong");
    g.add_conditional_edges(
        "pong",
        Arc::new(|values| {
            let c = values.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            if c >= 5 { vec![END.into()] } else { vec!["ping".into()] }
        }),
        None,
    );
    let app = Arc::new(g.compile(CompileConfig::default()).await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?);
    let out = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("final count = {}", out["count"]);
    Ok(())
}
