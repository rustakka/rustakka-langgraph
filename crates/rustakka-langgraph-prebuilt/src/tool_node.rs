//! `ToolNode` and `tools_condition` mirroring `langgraph.prebuilt`.
//!
//! Tools are arbitrary async closures `(args: Value) -> Value`. The node
//! consumes the latest message's `tool_calls` field, executes each tool in
//! parallel, and emits the `ToolMessage` results as `messages` updates.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustakka_langgraph_core::errors::{GraphError, GraphResult};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub args: Value,
}

pub type ToolFn =
    Arc<dyn Fn(Value) -> Pin<Box<dyn std::future::Future<Output = GraphResult<Value>> + Send>> + Send + Sync>;

pub struct Tool {
    pub name: String,
    pub description: String,
    pub func: ToolFn,
}

impl Tool {
    pub fn new<F, Fut>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = GraphResult<Value>> + Send + 'static,
    {
        let func: ToolFn = Arc::new(move |v| Box::pin(f(v)));
        Self { name: name.into(), description: description.into(), func }
    }
}

/// Configuration for [`ToolNode::into_node_with_options`].
#[derive(Debug, Clone)]
pub struct ToolNodeOptions {
    /// Maximum number of tool calls to run concurrently for a single node
    /// invocation. `1` falls back to sequential execution. Defaults to
    /// [`ToolNodeOptions::DEFAULT_PARALLELISM`] (8).
    pub parallelism: usize,
}

impl ToolNodeOptions {
    pub const DEFAULT_PARALLELISM: usize = 8;
}

impl Default for ToolNodeOptions {
    fn default() -> Self {
        Self { parallelism: Self::DEFAULT_PARALLELISM }
    }
}

pub struct ToolNode {
    tools: Arc<BTreeMap<String, Tool>>,
}

impl ToolNode {
    pub fn new(tools: impl IntoIterator<Item = Tool>) -> Self {
        let mut map = BTreeMap::new();
        for t in tools {
            map.insert(t.name.clone(), t);
        }
        Self { tools: Arc::new(map) }
    }

    /// Build a [`NodeKind`] using the default options (parallel fan-out).
    pub fn into_node(self) -> NodeKind {
        self.into_node_with_options(ToolNodeOptions::default())
    }

    /// Build a [`NodeKind`] honoring the provided options. Tool calls are
    /// dispatched through a `rustakka-streams` `Source::from_iter(calls)
    /// .map_async_unordered(parallelism, ..)` pipeline so they execute with
    /// bounded concurrency, and the collected `ToolMessage` payloads are
    /// re-ordered deterministically (by original call index) before being
    /// emitted on the `messages` channel.
    pub fn into_node_with_options(self, opts: ToolNodeOptions) -> NodeKind {
        let tools = self.tools.clone();
        let parallelism = opts.parallelism.max(1);
        NodeKind::from_fn(move |input: BTreeMap<String, Value>| {
            let tools = tools.clone();
            async move {
                let messages = input.get("messages").cloned().unwrap_or(Value::Array(vec![]));
                let last = messages
                    .as_array()
                    .and_then(|a| a.last().cloned())
                    .unwrap_or(Value::Null);
                let calls: Vec<ToolCall> = last
                    .get("tool_calls")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();

                let indexed = calls.into_iter().enumerate().collect::<Vec<_>>();
                let source = rustakka_streams::Source::from_iter(indexed).map_async_unordered(
                    parallelism,
                    move |(idx, c): (usize, ToolCall)| {
                        let tools = tools.clone();
                        async move { (idx, run_single_tool(tools, c).await) }
                    },
                );
                let mut results: Vec<(usize, GraphResult<Value>)> =
                    rustakka_streams::Sink::collect(source).await;
                results.sort_by_key(|(i, _)| *i);
                let mut tool_messages = Vec::with_capacity(results.len());
                for (_, res) in results {
                    tool_messages.push(res?);
                }
                let mut out = BTreeMap::new();
                out.insert("messages".into(), Value::Array(tool_messages));
                Ok(NodeOutput::Update(out))
            }
        })
    }
}

async fn run_single_tool(
    tools: Arc<BTreeMap<String, Tool>>,
    c: ToolCall,
) -> GraphResult<Value> {
    if let Some(t) = tools.get(&c.name) {
        let res = (t.func)(c.args.clone()).await.map_err(|e| GraphError::Node {
            node: "tools".into(),
            source: anyhow::anyhow!(e.to_string()),
        })?;
        Ok(json!({
            "id": c.id,
            "type": "tool",
            "name": c.name,
            "content": res,
        }))
    } else {
        Ok(json!({
            "id": c.id,
            "type": "tool",
            "name": c.name,
            "content": format!("error: unknown tool `{}`", c.name),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    fn tool_calls_input(calls: Vec<ToolCall>) -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        m.insert(
            "messages".into(),
            json!([{ "role": "assistant", "tool_calls": calls }]),
        );
        m
    }

    /// Parallelism > 1 must run overlapping tool calls concurrently:
    /// four 80ms sleeps with `parallelism=4` must finish well under
    /// the sequential 320ms lower bound.
    #[tokio::test]
    async fn parallel_tool_calls_beat_sequential_wall_clock() {
        let tool = Tool::new("sleep_tool", "sleep then echo", |v: Value| async move {
            tokio::time::sleep(Duration::from_millis(80)).await;
            Ok(v)
        });
        let node = ToolNode::new([tool])
            .into_node_with_options(ToolNodeOptions { parallelism: 4 });
        let calls = (0..4)
            .map(|i| ToolCall {
                id: format!("c{i}"),
                name: "sleep_tool".into(),
                args: json!({ "i": i }),
            })
            .collect();
        let start = Instant::now();
        let out = node.invoke(tool_calls_input(calls)).await.unwrap();
        let elapsed = start.elapsed();
        let NodeOutput::Update(update) = out else {
            panic!("expected update output")
        };
        let msgs = update.get("messages").and_then(|v| v.as_array()).cloned().unwrap();
        assert_eq!(msgs.len(), 4);
        assert!(
            elapsed < Duration::from_millis(280),
            "expected concurrent execution, got {:?}",
            elapsed
        );
    }

    /// Parallelism == 1 enforces sequential execution — four 60ms sleeps
    /// must take at least 4×60ms minus a small jitter margin.
    #[tokio::test]
    async fn parallelism_one_is_sequential() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        COUNTER.store(0, Ordering::SeqCst);
        let tool = Tool::new("s", "s", |_v: Value| async move {
            COUNTER.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(60)).await;
            Ok(json!(COUNTER.load(Ordering::SeqCst)))
        });
        let node = ToolNode::new([tool])
            .into_node_with_options(ToolNodeOptions { parallelism: 1 });
        let calls = (0..4)
            .map(|i| ToolCall { id: format!("c{i}"), name: "s".into(), args: json!({}) })
            .collect();
        let start = Instant::now();
        node.invoke(tool_calls_input(calls)).await.unwrap();
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(220), "got {:?}", elapsed);
    }

    /// Even with `map_async_unordered`, the node must emit tool messages
    /// in the same order as the original tool_calls array.
    #[tokio::test]
    async fn preserves_call_order_in_output() {
        let tool = Tool::new("echo", "echo", |v: Value| async move {
            let delay = v.get("delay_ms").and_then(|d| d.as_u64()).unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok(v)
        });
        let node = ToolNode::new([tool])
            .into_node_with_options(ToolNodeOptions { parallelism: 4 });
        // Reverse-gradient delays: first call sleeps longest, so if
        // ordering just followed completion we'd get [fourth, third, ...]
        let calls = vec![
            ToolCall { id: "a".into(), name: "echo".into(), args: json!({ "delay_ms": 60 }) },
            ToolCall { id: "b".into(), name: "echo".into(), args: json!({ "delay_ms": 40 }) },
            ToolCall { id: "c".into(), name: "echo".into(), args: json!({ "delay_ms": 20 }) },
            ToolCall { id: "d".into(), name: "echo".into(), args: json!({ "delay_ms": 0 }) },
        ];
        let NodeOutput::Update(update) = node.invoke(tool_calls_input(calls)).await.unwrap() else {
            panic!("expected update");
        };
        let msgs = update.get("messages").and_then(|v| v.as_array()).cloned().unwrap();
        let ids: Vec<String> = msgs
            .iter()
            .map(|m| m.get("id").and_then(|v| v.as_str()).unwrap().to_string())
            .collect();
        assert_eq!(ids, vec!["a", "b", "c", "d"]);
    }
}

/// Router predicate: "tools" if the last message has tool_calls, else END.
pub fn tools_condition(values: &BTreeMap<String, Value>) -> Vec<String> {
    let messages = values.get("messages").cloned().unwrap_or(Value::Array(vec![]));
    let last = messages.as_array().and_then(|a| a.last()).cloned().unwrap_or(Value::Null);
    let has_calls = last
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    if has_calls {
        vec!["tools".into()]
    } else {
        vec![rustakka_langgraph_core::graph::END.into()]
    }
}
