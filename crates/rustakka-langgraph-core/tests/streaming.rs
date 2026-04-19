//! End-to-end streaming parity: values/updates/custom/messages events flow
//! through the StreamBus to subscribers during a run.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use rustakka_langgraph_core::config::{RunnableConfig, StreamMode};
use rustakka_langgraph_core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::stream;
use rustakka_langgraph_core::state::DynamicState;
use rustakka_langgraph_core::stream::{current_writer, StreamEvent};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_emits_values_updates_custom_and_messages() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "a",
        NodeKind::from_fn(|_input| async move {
            if let Some(w) = current_writer() {
                w.custom(json!({"tick": 1}));
                w.message(json!({"role": "assistant", "content": "hi"}));
            }
            let mut m = BTreeMap::new();
            m.insert("x".into(), json!(1));
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_edge(START, "a");
    g.add_edge("a", END);
    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());

    let (mut rx, handle) = stream(
        app,
        BTreeMap::new(),
        RunnableConfig::default(),
        vec![
            StreamMode::Values,
            StreamMode::Updates,
            StreamMode::Custom,
            StreamMode::Messages,
        ],
    );

    let run_res = handle.await.unwrap().unwrap();
    assert_eq!(run_res.values["x"], json!(1));

    let mut saw_values = false;
    let mut saw_updates = false;
    let mut saw_custom = false;
    let mut saw_messages = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            StreamEvent::Values { .. } => saw_values = true,
            StreamEvent::Updates { node, .. } => {
                assert_eq!(node, "a");
                saw_updates = true;
            }
            StreamEvent::Custom { payload, .. } => {
                assert_eq!(payload, json!({"tick": 1}));
                saw_custom = true;
            }
            StreamEvent::Messages { message, .. } => {
                assert_eq!(message["content"], json!("hi"));
                saw_messages = true;
            }
            StreamEvent::Debug { .. } => {}
        }
    }
    assert!(saw_values, "expected Values event");
    assert!(saw_updates, "expected Updates event");
    assert!(saw_custom, "expected Custom event");
    assert!(saw_messages, "expected Messages event");
}
