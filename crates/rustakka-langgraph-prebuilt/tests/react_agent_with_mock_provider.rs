//! End-to-end test: `create_react_agent` driven by a real `MockChatModel`
//! via the `chat_model_fn` adapter. Exercises the tool-call loop with the
//! full Message <-> JSON shape conversions.

#![cfg(feature = "providers")]

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::runner::invoke_dynamic;
use rustakka_langgraph_prebuilt::{
    chat_model_fn, create_react_agent, InvocationMode, ReactAgentOptions, Tool,
};
use rustakka_langgraph_providers::prelude::*;
use rustakka_langgraph_providers::types::message::ToolCallRequest;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn react_agent_with_mock_chat_model_invoke() {
    let mock = MockChatModel::new(vec![
        Message::ai_with_tool_calls(
            "",
            vec![ToolCallRequest {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 40, "b": 2}),
            }],
        ),
        Message::ai("the answer is 42"),
    ]);
    let model_fn = chat_model_fn(
        Arc::new(mock),
        CallOptions::default(),
        InvocationMode::Invoke,
    );

    let add_tool = Tool::new("add", "adds two numbers", |args: Value| async move {
        let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
        let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
        Ok(json!(a + b))
    });

    let app = create_react_agent(
        model_fn,
        vec![add_tool],
        ReactAgentOptions {
            recursion_limit: Some(10),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let app = Arc::new(app);
    let mut input = BTreeMap::new();
    input.insert(
        "messages".into(),
        json!([{"role": "user", "content": "40+2?"}]),
    );
    let out = invoke_dynamic(app, input, RunnableConfig::default()).await.unwrap();
    let msgs = out["messages"].as_array().unwrap();
    let last = msgs.last().unwrap();
    assert_eq!(last["content"], json!("the answer is 42"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn react_agent_streaming_mode_assembles_chunks() {
    // MockChatModel::stream splits text on whitespace — one chunk per word.
    // Two turns: first emits a tool call, second emits plain text.
    let mock = MockChatModel::new(vec![
        Message::ai_with_tool_calls(
            "",
            vec![ToolCallRequest {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 1, "b": 2}),
            }],
        ),
        Message::ai("final answer three"),
    ]);
    let model_fn = chat_model_fn(
        Arc::new(mock),
        CallOptions::default(),
        InvocationMode::Stream,
    );

    let add_tool = Tool::new("add", "adds two numbers", |args: Value| async move {
        let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
        let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
        Ok(json!(a + b))
    });

    let app = create_react_agent(
        model_fn,
        vec![add_tool],
        ReactAgentOptions {
            recursion_limit: Some(10),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let app = Arc::new(app);
    let mut input = BTreeMap::new();
    input.insert(
        "messages".into(),
        json!([{"role": "user", "content": "compute"}]),
    );
    let out = invoke_dynamic(app, input, RunnableConfig::default()).await.unwrap();
    let msgs = out["messages"].as_array().unwrap();
    let last = msgs.last().unwrap();
    // MockChatModel::stream splits on whitespace and drops the separators, so
    // the assembled text is the concatenation of the tokens. What matters here
    // is that streaming correctly reaches the final step and stops the loop.
    assert_eq!(last["content"], json!("finalanswerthree"));
    assert_eq!(last["role"], json!("assistant"));
}
