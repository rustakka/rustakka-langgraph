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
            // v2 `astream_events` kinds — tolerated in this fixture.
            StreamEvent::OnChainStart { .. }
            | StreamEvent::OnChainEnd { .. }
            | StreamEvent::OnChatModelStream { .. }
            | StreamEvent::OnToolStart { .. }
            | StreamEvent::OnToolEnd { .. } => {}
        }
    }
    assert!(saw_values, "expected Values event");
    assert!(saw_updates, "expected Updates event");
    assert!(saw_custom, "expected Custom event");
    assert!(saw_messages, "expected Messages event");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn astream_events_v2_kinds_are_emitted() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "tool_node",
        NodeKind::from_fn(|_| async move {
            if let Some(w) = current_writer() {
                w.tool_start("search", json!({"q": "cats"}));
                w.chat_model_chunk(json!({"delta": "he"}));
                w.chat_model_chunk(json!({"delta": "llo"}));
                w.tool_end("search", json!({"hits": 3}));
            }
            Ok(NodeOutput::Update(BTreeMap::new()))
        }),
    )
    .unwrap();
    g.add_edge(START, "tool_node");
    g.add_edge("tool_node", END);
    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());

    let (mut rx, handle) = stream(
        app,
        BTreeMap::new(),
        RunnableConfig::default(),
        vec![StreamMode::Events],
    );
    let _ = handle.await.unwrap().unwrap();

    let mut saw_tool_start = false;
    let mut saw_tool_end = false;
    let mut saw_chain_start = false;
    let mut saw_chain_end = false;
    let mut saw_chat_stream = 0u8;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            StreamEvent::OnToolStart { tool, .. } => {
                assert_eq!(tool, "search");
                saw_tool_start = true;
            }
            StreamEvent::OnToolEnd { tool, .. } => {
                assert_eq!(tool, "search");
                saw_tool_end = true;
            }
            StreamEvent::OnChainStart { .. } => saw_chain_start = true,
            StreamEvent::OnChainEnd { .. } => saw_chain_end = true,
            StreamEvent::OnChatModelStream { .. } => saw_chat_stream += 1,
            _ => {}
        }
    }
    assert!(saw_tool_start);
    assert!(saw_tool_end);
    assert!(saw_chain_start);
    assert!(saw_chain_end);
    assert_eq!(saw_chat_stream, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subgraph_events_are_namespaced() {
    // Child: emits a custom event from a node named `inner`.
    let mut child = StateGraph::<DynamicState>::new();
    child
        .add_node(
            "inner",
            NodeKind::from_fn(|_| async move {
                if let Some(w) = current_writer() {
                    w.custom(json!("from-child"));
                }
                Ok(NodeOutput::Update(BTreeMap::new()))
            }),
        )
        .unwrap();
    child.add_edge(START, "inner");
    child.add_edge("inner", END);
    let compiled_child = Arc::new(child.compile(CompileConfig::default()).await.unwrap());

    // Parent embeds child under node `bridge`.
    let mut parent = StateGraph::<DynamicState>::new();
    parent
        .add_node(
            "bridge",
            NodeKind::Subgraph(compiled_child.clone().as_subgraph_invoker()),
        )
        .unwrap();
    parent.add_edge(START, "bridge");
    parent.add_edge("bridge", END);
    let app = Arc::new(parent.compile(CompileConfig::default()).await.unwrap());

    let (mut rx, handle) = stream(
        app,
        BTreeMap::new(),
        RunnableConfig::default(),
        vec![StreamMode::Custom],
    );
    let _ = handle.await.unwrap().unwrap();

    let mut namespaced = None;
    while let Ok(ev) = rx.try_recv() {
        if let StreamEvent::Custom { namespace, payload, .. } = ev {
            if !namespace.is_empty() {
                namespaced = Some((namespace, payload));
                break;
            }
        }
    }
    let (ns, payload) = namespaced.expect("expected a namespaced Custom event from child");
    assert_eq!(ns, vec!["bridge".to_string()]);
    assert_eq!(payload, json!("from-child"));
}
