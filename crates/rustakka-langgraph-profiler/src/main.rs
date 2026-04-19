//! Cross-runtime profiler. Emits the same JSON schema as rustakka-profiler so
//! results can be merged side-by-side with actor benchmarks.
//!
//! Scenarios:
//!   - `invoke`           — single small graph end-to-end
//!   - `stream`           — streaming run with N supersteps
//!   - `fanout`           — Send/fan-out across W workers
//!   - `checkpoint-heavy` — large state, MemorySaver each step
//!
//! Usage:
//!   cargo run --release -p rustakka-langgraph-profiler -- --scenario invoke --iterations 100

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use rustakka_langgraph_checkpoint::{CheckpointerHookAdapter, MemorySaver};
use rustakka_langgraph_core::config::{RunnableConfig, StreamMode};
use rustakka_langgraph_core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::{invoke_dynamic, invoke_with_checkpointer, stream as stream_run};
use rustakka_langgraph_core::state::DynamicState;

#[derive(Debug, Serialize)]
struct ScenarioResult {
    scenario: String,
    iterations: u32,
    elapsed_ms: f64,
    p50_us: f64,
    p99_us: f64,
}

fn parse_args() -> (String, u32) {
    let args: Vec<String> = std::env::args().collect();
    let mut scenario = "invoke".to_string();
    let mut iters: u32 = 100;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scenario" => { scenario = args[i + 1].clone(); i += 2; }
            "--iterations" => { iters = args[i + 1].parse().unwrap_or(100); i += 2; }
            _ => i += 1,
        }
    }
    (scenario, iters)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let (scenario, iterations) = parse_args();
    let res = match scenario.as_str() {
        "invoke" => bench_invoke(iterations).await?,
        "fanout" => bench_fanout(iterations).await?,
        "stream" => bench_stream(iterations).await?,
        "checkpoint-heavy" => bench_checkpoint_heavy(iterations).await?,
        other => anyhow::bail!("unknown scenario `{other}`"),
    };
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

async fn bench_invoke(iters: u32) -> Result<ScenarioResult> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node("a", NodeKind::from_fn(|_| async {
        let mut m = BTreeMap::new();
        m.insert("x".into(), json!(1));
        Ok(NodeOutput::Update(m))
    }))
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.add_edge(START, "a");
    g.add_edge("a", END);
    let app = Arc::new(g.compile(CompileConfig::default()).await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?);
    let mut samples_us: Vec<u128> = Vec::with_capacity(iters as usize);
    let total = Instant::now();
    for _ in 0..iters {
        let t = Instant::now();
        let _ = invoke_dynamic(app.clone(), BTreeMap::new(), RunnableConfig::default()).await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        samples_us.push(t.elapsed().as_micros());
    }
    Ok(summarize("invoke", iters, total.elapsed().as_secs_f64() * 1000.0, &samples_us))
}

async fn bench_fanout(iters: u32) -> Result<ScenarioResult> {
    let mut g = StateGraph::<DynamicState>::new();
    for i in 0..8u32 {
        let name = format!("w{i}");
        g.add_node(&name, NodeKind::from_fn(move |_| async move {
            let mut m = BTreeMap::new();
            m.insert("count".into(), json!(1));
            Ok(NodeOutput::Update(m))
        })).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        g.add_edge(START, &name);
        g.add_edge(&name, END);
    }
    let app = Arc::new(g.compile(CompileConfig::default()).await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?);
    let mut samples_us = Vec::with_capacity(iters as usize);
    let total = Instant::now();
    for _ in 0..iters {
        let t = Instant::now();
        let _ = invoke_dynamic(app.clone(), BTreeMap::new(), RunnableConfig::default()).await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        samples_us.push(t.elapsed().as_micros());
    }
    Ok(summarize("fanout", iters, total.elapsed().as_secs_f64() * 1000.0, &samples_us))
}

async fn bench_stream(iters: u32) -> Result<ScenarioResult> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "a",
        NodeKind::from_fn(|_| async {
            let mut m = BTreeMap::new();
            m.insert("x".into(), json!(1));
            Ok(NodeOutput::Update(m))
        }),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.add_node(
        "b",
        NodeKind::from_fn(|input| async move {
            let cur = input.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut m = BTreeMap::new();
            m.insert("x".into(), json!(cur + 1));
            Ok(NodeOutput::Update(m))
        }),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.add_edge(START, "a");
    g.add_edge("a", "b");
    g.add_edge("b", END);
    let app = Arc::new(
        g.compile(CompileConfig::default()).await.map_err(|e| anyhow::anyhow!(e.to_string()))?,
    );
    let mut samples = Vec::with_capacity(iters as usize);
    let total = Instant::now();
    for _ in 0..iters {
        let t = Instant::now();
        let (mut rx, h) = stream_run(
            app.clone(),
            BTreeMap::new(),
            RunnableConfig::default(),
            vec![StreamMode::Values, StreamMode::Updates],
        );
        while rx.recv().await.is_some() {}
        let _ = h.await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
        samples.push(t.elapsed().as_micros());
    }
    Ok(summarize("stream", iters, total.elapsed().as_secs_f64() * 1000.0, &samples))
}

async fn bench_checkpoint_heavy(iters: u32) -> Result<ScenarioResult> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "grow",
        NodeKind::from_fn(|input| async move {
            let mut arr = input
                .get("blob")
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            for i in 0..256 {
                arr.push(json!(i));
            }
            let mut m = BTreeMap::new();
            m.insert("blob".into(), json!(arr));
            Ok(NodeOutput::Update(m))
        }),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.add_edge(START, "grow");
    g.add_edge("grow", END);
    let app = Arc::new(
        g.compile(CompileConfig::default()).await.map_err(|e| anyhow::anyhow!(e.to_string()))?,
    );
    let saver = Arc::new(MemorySaver::new());
    let hook = CheckpointerHookAdapter::new(saver);
    let mut samples = Vec::with_capacity(iters as usize);
    let total = Instant::now();
    for i in 0..iters {
        let cfg = RunnableConfig::with_thread(format!("thr-{i}"));
        let t = Instant::now();
        let _ = invoke_with_checkpointer(app.clone(), BTreeMap::new(), cfg, hook.clone())
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        samples.push(t.elapsed().as_micros());
    }
    Ok(summarize(
        "checkpoint-heavy",
        iters,
        total.elapsed().as_secs_f64() * 1000.0,
        &samples,
    ))
}

fn summarize(name: &str, iters: u32, total_ms: f64, samples: &[u128]) -> ScenarioResult {
    let mut s: Vec<u128> = samples.to_vec();
    s.sort_unstable();
    let p = |q: f64| -> f64 {
        if s.is_empty() { return 0.0; }
        let idx = ((s.len() - 1) as f64 * q).round() as usize;
        s[idx] as f64
    };
    ScenarioResult {
        scenario: name.into(),
        iterations: iters,
        elapsed_ms: total_ms,
        p50_us: p(0.50),
        p99_us: p(0.99),
    }
}
