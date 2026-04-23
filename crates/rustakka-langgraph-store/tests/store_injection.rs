//! End-to-end: attach an `InMemoryStore` to a graph run and prove nodes
//! can reach it via `rustakka_langgraph_core::context::get_store`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::context::get_store;
use rustakka_langgraph_core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::invoke_with_store;
use rustakka_langgraph_core::state::DynamicState;

use rustakka_langgraph_store::{store_accessor, InMemoryStore};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_can_read_and_write_via_current_store() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "writer",
        NodeKind::from_fn(|_| async move {
            let store = get_store().expect("store should be injected");
            store
                .put(&["prefs".into()], "theme", json!("dark"), None)
                .await
                .unwrap();
            Ok(NodeOutput::Update(BTreeMap::new()))
        }),
    )
    .unwrap();
    g.add_node(
        "reader",
        NodeKind::from_fn(|_| async move {
            let store = get_store().expect("store should be injected");
            let v = store
                .get(&["prefs".into()], "theme")
                .await
                .unwrap()
                .expect("writer should have set the key");
            let mut m = BTreeMap::new();
            m.insert("theme".into(), v);
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_edge(START, "writer");
    g.add_edge("writer", "reader");
    g.add_edge("reader", END);

    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
    let store = store_accessor(Arc::new(InMemoryStore::new()));
    let res = invoke_with_store(
        app,
        BTreeMap::new(),
        RunnableConfig::default(),
        None,
        store,
    )
    .await
    .unwrap();
    assert_eq!(res.values["theme"], json!("dark"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_store_returns_none_when_not_attached() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "check",
        NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("has_store".into(), json!(get_store().is_some()));
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_edge(START, "check");
    g.add_edge("check", END);
    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
    let out = rustakka_langgraph_core::runner::invoke_dynamic(
        app,
        BTreeMap::new(),
        RunnableConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(out["has_store"], json!(false));
}
