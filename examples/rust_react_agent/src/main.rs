//! Minimal pure-Rust ReAct agent driven by a fake LLM that "decides" to call
//! a calculator tool exactly once before stopping.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::runner::invoke_dynamic;
use rustakka_langgraph_prebuilt::{create_react_agent, ReactAgentOptions, Tool};
use serde_json::json;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let calc = Tool::new("calc", "Add two numbers", |args| async move {
        let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
        let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
        Ok(json!({"sum": a + b}))
    });

    let model = Arc::new(|messages: Vec<serde_json::Value>, _sys: Option<String>| {
        Box::pin(async move {
            // first call -> request a tool, second call -> finalize
            let last_is_tool = messages
                .last()
                .and_then(|m| m.get("type"))
                .and_then(|t| t.as_str())
                == Some("tool");
            if last_is_tool {
                Ok(json!({"id": "ai-final", "type": "ai", "content": "done"}))
            } else {
                Ok(json!({
                    "id": "ai-1", "type": "ai", "content": "calling tool",
                    "tool_calls": [{"id": "t1", "name": "calc", "args": {"a": 2, "b": 3}}]
                }))
            }
        })
            as std::pin::Pin<Box<dyn std::future::Future<Output = rustakka_langgraph_core::errors::GraphResult<serde_json::Value>> + Send>>
    });

    let app = Arc::new(create_react_agent(model, vec![calc], ReactAgentOptions::default()).await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?);
    let out = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("messages: {}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
