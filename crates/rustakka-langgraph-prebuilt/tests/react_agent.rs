//! `create_react_agent` + `ToolNode` smoke test using a fake LLM.
//!
//! The fake model emits a tool call on first turn and a final answer on the
//! second turn, exercising the `agent → tools → agent → END` cycle and the
//! `tools_condition` router.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::runner::invoke_dynamic;
use rustakka_langgraph_prebuilt::{
    create_react_agent, tools_condition, ReactAgentOptions, Tool, ToolNode,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn react_agent_tool_loop_terminates() {
    let turns = Arc::new(AtomicUsize::new(0));
    let t = turns.clone();
    let model: rustakka_langgraph_prebuilt::react_agent::ModelFn = Arc::new(move |_msgs, _sys| {
        let t = t.clone();
        Box::pin(async move {
            let n = t.fetch_add(1, Ordering::SeqCst);
            let out = if n == 0 {
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "name": "add", "args": {"a": 40, "b": 2}}
                    ]
                })
            } else {
                json!({"role": "assistant", "content": "the answer is 42"})
            };
            Ok::<Value, rustakka_langgraph_core::errors::GraphError>(out)
        })
    });

    let add_tool = Tool::new("add", "add two numbers", |args: Value| async move {
        let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
        let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
        Ok(json!(a + b))
    });

    let app = create_react_agent(
        model,
        vec![add_tool],
        ReactAgentOptions { recursion_limit: Some(10), ..Default::default() },
    )
    .await
    .unwrap();

    let app = Arc::new(app);
    let mut input = BTreeMap::new();
    input.insert("messages".into(), json!([{"role": "user", "content": "40+2?"}]));
    let out = invoke_dynamic(app, input, RunnableConfig::default()).await.unwrap();
    let msgs = out["messages"].as_array().unwrap();
    let last = msgs.last().unwrap();
    assert_eq!(last["content"], json!("the answer is 42"));
    assert!(turns.load(Ordering::SeqCst) >= 2, "model should be called at least twice");
}

#[test]
fn tools_condition_detects_calls() {
    let mut st = BTreeMap::new();
    st.insert(
        "messages".into(),
        json!([{"role": "assistant", "tool_calls": [{"id": "1", "name": "x", "args": {}}]}]),
    );
    assert_eq!(tools_condition(&st), vec!["tools".to_string()]);

    st.insert("messages".into(), json!([{"role": "assistant", "content": "done"}]));
    let r = tools_condition(&st);
    assert_eq!(r, vec![rustakka_langgraph_core::graph::END.to_string()]);
}

#[tokio::test]
async fn tool_node_emits_tool_messages() {
    let t = Tool::new("echo", "echoes args", |args: Value| async move { Ok(args) });
    let node = ToolNode::new(vec![t]).into_node();
    let mut input = BTreeMap::new();
    input.insert(
        "messages".into(),
        json!([{"tool_calls": [{"id": "c1", "name": "echo", "args": {"ok": true}}]}]),
    );
    let out = node.invoke(input).await.unwrap();
    match out {
        rustakka_langgraph_core::node::NodeOutput::Update(m) => {
            let msgs = m["messages"].as_array().unwrap();
            assert_eq!(msgs[0]["content"], json!({"ok": true}));
        }
        _ => panic!("expected update"),
    }
}
