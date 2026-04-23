//! Node-level `CachePolicy` integration tests.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::graph::{
    CachePolicy, CompileConfig, StateGraph, END, START,
};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::invoke_dynamic;
use rustakka_langgraph_core::state::DynamicState;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn second_invocation_replays_cached_update() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_inner = calls.clone();

    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "work",
        NodeKind::from_fn(move |_| {
            let c = calls_inner.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                let mut m = BTreeMap::new();
                m.insert("n".into(), json!(42));
                Ok(NodeOutput::Update(m))
            }
        }),
    )
    .unwrap();
    g.add_edge(START, "work");
    g.add_edge("work", END);

    // Cache key ignores input (empty), so every invocation hits the same key.
    let policy = CachePolicy {
        key_func: Arc::new(|_: &BTreeMap<String, Value>| "const".to_string()),
        ttl_seconds: None,
    };
    g.set_cache_policy("work", policy);

    let compiled = Arc::new(g.compile(CompileConfig::default()).await.unwrap());

    let r1 = invoke_dynamic(compiled.clone(), BTreeMap::new(), RunnableConfig::default())
        .await
        .unwrap();
    let r2 = invoke_dynamic(compiled.clone(), BTreeMap::new(), RunnableConfig::default())
        .await
        .unwrap();

    assert_eq!(r1["n"], json!(42));
    assert_eq!(r2["n"], json!(42));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cached hit should skip the second invocation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ttl_zero_seconds_expires_immediately() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_inner = calls.clone();

    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "work",
        NodeKind::from_fn(move |_| {
            let c = calls_inner.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(NodeOutput::Update(BTreeMap::new()))
            }
        }),
    )
    .unwrap();
    g.add_edge(START, "work");
    g.add_edge("work", END);
    g.set_cache_policy("work", CachePolicy::new(Some(0)));

    let compiled = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
    invoke_dynamic(compiled.clone(), BTreeMap::new(), RunnableConfig::default())
        .await
        .unwrap();
    // Wait long enough for the 0-second TTL to elapse.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    invoke_dynamic(compiled.clone(), BTreeMap::new(), RunnableConfig::default())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
