//! End-to-end test: compile a graph, run it with SqliteSaver, verify the
//! checkpoint row is persisted and restorable.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use rustakka_langgraph_checkpoint::{CheckpointerHookAdapter, MemorySaver};
use rustakka_langgraph_checkpoint_sqlite::SqliteSaver;
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::invoke_with_checkpointer;
use rustakka_langgraph_core::state::DynamicState;

async fn build_app() -> Arc<rustakka_langgraph_core::graph::CompiledStateGraph> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "inc",
        NodeKind::from_fn(|input: BTreeMap<String, serde_json::Value>| async move {
            let c = input.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut m = BTreeMap::new();
            m.insert("count".into(), json!(c + 1));
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_edge(START, "inc");
    g.add_edge("inc", END);
    Arc::new(g.compile(CompileConfig::default()).await.unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_checkpoint_resumes_from_latest() {
    let app = build_app().await;
    let saver = Arc::new(MemorySaver::new());
    let hook = CheckpointerHookAdapter::new(saver.clone());
    let cfg = RunnableConfig::with_thread("t1");
    let r1 = invoke_with_checkpointer(app.clone(), BTreeMap::new(), cfg.clone(), hook.clone())
        .await
        .unwrap();
    assert_eq!(r1.values["count"], json!(1));

    // Second call resumes from the checkpoint: count is already 1 -> becomes 2.
    let r2 = invoke_with_checkpointer(app, BTreeMap::new(), cfg, hook).await.unwrap();
    assert_eq!(r2.values["count"], json!(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sqlite_checkpoint_persists_across_instances() {
    let app = build_app().await;
    // Use an in-memory SQLite instance with explicit shared cache so multiple
    // pool connections see the same data.
    let saver = Arc::new(SqliteSaver::in_memory().await.unwrap());
    let hook = CheckpointerHookAdapter::new(saver.clone());
    let cfg = RunnableConfig::with_thread("t1");
    let r1 = invoke_with_checkpointer(app.clone(), BTreeMap::new(), cfg.clone(), hook.clone())
        .await
        .unwrap();
    assert_eq!(r1.values["count"], json!(1));
    // A different invocation against the same saver should resume state.
    let r2 = invoke_with_checkpointer(app, BTreeMap::new(), cfg, hook).await.unwrap();
    assert_eq!(r2.values["count"], json!(2));
}
