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

    pub fn into_node(self) -> NodeKind {
        let tools = self.tools.clone();
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
                let mut tool_messages = Vec::new();
                for c in calls {
                    if let Some(t) = tools.get(&c.name) {
                        let res = (t.func)(c.args.clone())
                            .await
                            .map_err(|e| GraphError::Node {
                                node: "tools".into(),
                                source: anyhow::anyhow!(e.to_string()),
                            })?;
                        tool_messages.push(json!({
                            "id": c.id,
                            "type": "tool",
                            "name": c.name,
                            "content": res,
                        }));
                    } else {
                        tool_messages.push(json!({
                            "id": c.id,
                            "type": "tool",
                            "name": c.name,
                            "content": format!("error: unknown tool `{}`", c.name),
                        }));
                    }
                }
                let mut out = BTreeMap::new();
                out.insert("messages".into(), Value::Array(tool_messages));
                Ok(NodeOutput::Update(out))
            }
        })
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
