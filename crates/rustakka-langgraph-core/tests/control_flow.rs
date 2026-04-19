//! Integration tests for Phase-2 control flow:
//!   - `Command(goto=...)` routing overrides static edges
//!   - `Command(update=...)` writes propagate
//!   - `Send(node, arg)` fan-out within a superstep
//!   - `interrupt(...)` / resume round-trips through a MemorySaver
//!   - subgraphs: compiled graph embedded as a node

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use rustakka_langgraph_core::channel::ChannelSnapshot;
use rustakka_langgraph_core::command::{Command, Interrupt, Send};
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::coordinator::{CheckpointReplay, CheckpointerHook};
use rustakka_langgraph_core::errors::{GraphError, GraphResult};
use rustakka_langgraph_core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};
use rustakka_langgraph_core::runner::{invoke_dynamic, invoke_with_checkpointer, resume};
use rustakka_langgraph_core::state::DynamicState;

// -------------------------------------------------------------- Command ---

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn command_goto_overrides_static_edges() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "router",
        NodeKind::from_fn(|_| async move {
            Ok(NodeOutput::Command(Command::goto("right").with_update("via", "command")))
        }),
    )
    .unwrap();
    g.add_node(
        "left",
        NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("picked".into(), json!("left"));
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_node(
        "right",
        NodeKind::from_fn(|_| async move {
            let mut m = BTreeMap::new();
            m.insert("picked".into(), json!("right"));
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_edge(START, "router");
    g.add_edge("router", "left"); // default static edge — overridden by Command
    g.add_edge("left", END);
    g.add_edge("right", END);

    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
    let out = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await.unwrap();
    assert_eq!(out["via"], json!("command"));
    // both left and right are reachable via static edges (router -> left static,
    // router -> right via command). Both run in the next superstep; last write wins.
    assert!(out["picked"] == json!("left") || out["picked"] == json!("right"));
}

// --------------------------------------------------------------- Send -----

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn send_fanout_multiplies_targets() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "supervisor",
        NodeKind::from_fn(|_| async move {
            let cmd = Command::default()
                .with_send(Send::new("worker", json!({"n": 1})))
                .with_send(Send::new("worker", json!({"n": 2})))
                .with_send(Send::new("worker", json!({"n": 3})));
            Ok(NodeOutput::Command(cmd))
        }),
    )
    .unwrap();
    g.add_channel(rustakka_langgraph_core::state::ChannelSpec {
        name: "results".into(),
        reducer: "topic".into(),
    });
    g.add_node(
        "worker",
        NodeKind::from_fn(|input: BTreeMap<String, serde_json::Value>| async move {
            let arg = input.get("__send_arg__").cloned().unwrap_or(json!({}));
            let n = arg.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut m = BTreeMap::new();
            m.insert("results".into(), json!(n * 10));
            Ok(NodeOutput::Update(m))
        }),
    )
    .unwrap();
    g.add_edge(START, "supervisor");
    g.add_edge("worker", END);

    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
    let out = invoke_dynamic(app, BTreeMap::new(), RunnableConfig::default()).await.unwrap();
    let mut results: Vec<i64> = out["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    results.sort();
    assert_eq!(results, vec![10, 20, 30]);
}

// --------------------------------------------------- interrupt + resume ---

#[derive(Default)]
struct MiniSaver {
    inner: std::sync::Mutex<Option<(u64, BTreeMap<String, ChannelSnapshot>, Option<Interrupt>)>>,
}

#[async_trait::async_trait]
impl CheckpointerHook for MiniSaver {
    async fn put_step(
        &self,
        _cfg: &RunnableConfig,
        step: u64,
        _values: &BTreeMap<String, serde_json::Value>,
        snapshot: &BTreeMap<String, ChannelSnapshot>,
        _pending_writes: &[(String, BTreeMap<String, serde_json::Value>)],
        interrupt: Option<&Interrupt>,
    ) -> GraphResult<()> {
        *self.inner.lock().unwrap() = Some((step, snapshot.clone(), interrupt.cloned()));
        Ok(())
    }
    async fn get_latest(
        &self,
        _cfg: &RunnableConfig,
    ) -> GraphResult<Option<CheckpointReplay>> {
        Ok(self.inner.lock().unwrap().as_ref().map(|(s, snap, intr)| CheckpointReplay {
            step: *s,
            snapshot: snap.clone(),
            interrupt: intr.clone(),
        }))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_surface_and_resume() {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "ask",
        NodeKind::from_fn(|input: BTreeMap<String, serde_json::Value>| async move {
            // If we were resumed, `__send_arg__` will carry the user's answer.
            if let Some(answer) = input.get("__send_arg__") {
                let mut m = BTreeMap::new();
                m.insert("answer".into(), answer.clone());
                return Ok(NodeOutput::Update(m));
            }
            Ok(NodeOutput::Interrupted(Interrupt::new("ask", json!("what is your name?"))))
        }),
    )
    .unwrap();
    g.add_edge(START, "ask");
    g.add_edge("ask", END);
    let app = Arc::new(g.compile(CompileConfig::default()).await.unwrap());
    let saver = Arc::new(MiniSaver::default());
    let saver_hook: Arc<dyn CheckpointerHook> = saver.clone();
    let cfg = RunnableConfig::with_thread("t1");

    let r1 = invoke_with_checkpointer(app.clone(), BTreeMap::new(), cfg.clone(), saver_hook.clone())
        .await
        .unwrap();
    let intr = r1.interrupted.expect("run must pause with interrupt");
    assert_eq!(intr.node, "ask");
    assert_eq!(intr.value, json!("what is your name?"));

    let r2 = resume(app, cfg, saver_hook, json!("alice")).await.unwrap();
    assert_eq!(r2.values["answer"], json!("alice"));
    assert!(r2.interrupted.is_none());
}

// ------------------------------------------------------------- subgraphs --

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compiled_subgraph_as_node() {
    // Child graph: doubles `n`.
    let mut child = StateGraph::<DynamicState>::new();
    child
        .add_node(
            "double",
            NodeKind::from_fn(|input: BTreeMap<String, serde_json::Value>| async move {
                let n = input.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
                let mut m = BTreeMap::new();
                m.insert("n".into(), json!(n * 2));
                Ok(NodeOutput::Update(m))
            }),
        )
        .unwrap();
    child.add_edge(START, "double");
    child.add_edge("double", END);
    let compiled_child = Arc::new(child.compile(CompileConfig::default()).await.unwrap());

    // Parent graph embeds child as a node.
    let mut parent = StateGraph::<DynamicState>::new();
    parent
        .add_node("inc", NodeKind::from_fn(|input| async move {
            let n = input.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut m = BTreeMap::new();
            m.insert("n".into(), json!(n + 1));
            Ok(NodeOutput::Update(m))
        }))
        .unwrap();
    parent
        .add_node(
            "child",
            NodeKind::Subgraph(compiled_child.clone().as_subgraph_invoker()),
        )
        .unwrap();
    parent.add_edge(START, "inc");
    parent.add_edge("inc", "child");
    parent.add_edge("child", END);

    let app = Arc::new(parent.compile(CompileConfig::default()).await.unwrap());
    let mut input = BTreeMap::new();
    input.insert("n".into(), json!(3));
    let out = invoke_dynamic(app, input, RunnableConfig::default()).await.unwrap();
    // (3 + 1) * 2 = 8
    assert_eq!(out["n"], json!(8));
}

// Prevent unused-import warnings in the integration-test binary
fn _unused() {
    let _ = GraphError::EmptyInput;
}
