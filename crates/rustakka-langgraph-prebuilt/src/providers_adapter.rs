//! Bridge between [`rustakka_langgraph_providers::ChatModel`] and the
//! [`crate::react_agent::ModelFn`] callback shape used by [`create_react_agent`].
//!
//! This module lives behind the `providers` feature so the rest of
//! `rustakka-langgraph-prebuilt` has no hard dependency on any LLM provider.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use rustakka_langgraph_core::errors::{GraphError, GraphResult};
use rustakka_langgraph_core::stream::current_writer;

use rustakka_langgraph_providers::prelude::*;

use crate::react_agent::ModelFn;

/// How to drive the underlying provider for each turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvocationMode {
    /// Single-shot: one `invoke` call per turn.
    Invoke,
    /// Streaming: call `stream` and also publish each chunk to the currently
    /// installed `StreamWriter` as a `Messages` event.
    Stream,
}

impl Default for InvocationMode {
    fn default() -> Self {
        InvocationMode::Invoke
    }
}

/// Build a [`ModelFn`] backed by a [`ChatModel`], suitable for
/// [`create_react_agent`](crate::react_agent::create_react_agent).
///
/// Graph state stores messages as plain JSON objects with the shape
/// `{"role", "content", "tool_calls": [{"id", "name", "args"}]}`. This adapter
/// handles bidirectional conversion between that shape and the provider's
/// unified [`Message`] type.
pub fn chat_model_fn(
    model: Arc<dyn ChatModel>,
    options: CallOptions,
    mode: InvocationMode,
) -> ModelFn {
    Arc::new(move |messages: Vec<Value>, system: Option<String>| {
        let model = model.clone();
        let opts = options.clone();
        Box::pin(async move {
            let mut msgs: Vec<Message> = Vec::with_capacity(messages.len() + 1);
            if let Some(s) = system {
                msgs.push(Message::system(s));
            }
            for v in messages {
                match json_to_message(&v) {
                    Ok(m) => msgs.push(m),
                    Err(e) => {
                        return Err(GraphError::Node {
                            node: "agent".into(),
                            source: anyhow::anyhow!(e),
                        })
                    }
                }
            }

            let reply = match mode {
                InvocationMode::Invoke => model
                    .invoke(&msgs, &opts)
                    .await
                    .map_err(|e| GraphError::Node {
                        node: "agent".into(),
                        source: anyhow::anyhow!(e.to_string()),
                    })?,
                InvocationMode::Stream => {
                    stream_into_message(model.clone(), msgs.clone(), opts.clone()).await?
                }
            };

            Ok(message_to_json(&reply))
        }) as _
    })
}

// ---------------------------------------------------------------------------
// Streaming: assemble chunks into a Message while publishing to StreamBus.
// ---------------------------------------------------------------------------

async fn stream_into_message(
    model: Arc<dyn ChatModel>,
    msgs: Vec<Message>,
    opts: CallOptions,
) -> GraphResult<Message> {
    let writer = current_writer();
    // Express the chunk stream as a rustakka-streams Source: side-effect
    // publishing happens through `wire_tap` (one `chat_model_chunk`
    // StreamEvent per successful chunk), and the final `Message` is
    // assembled by running the source through a `Sink::collect`. This
    // replaces the hand-rolled `while let Some(..) = stream.next()`
    // loop while preserving the public behavior.
    let writer_tap = writer.clone();
    let source = chat_model_stream_source(model, msgs, opts).wire_tap(move |chunk| {
        if let (Some(w), Ok(c)) = (writer_tap.as_ref(), chunk.as_ref()) {
            let payload = json!({
                "text": c.text,
                "tool_call_chunks": c.tool_call_chunks,
                "metadata": c.metadata,
            });
            w.chat_model_chunk(payload);
        }
    });
    let chunks: Vec<Result<GenerationChunk, ProviderError>> =
        rustakka_streams::Sink::collect(source).await;

    let mut text = String::new();
    // Keyed by chunk.index, holds (id, name, arguments buffer).
    let mut tool_buffers: BTreeMap<usize, PartialTool> = BTreeMap::new();

    for chunk_res in chunks {
        let chunk = chunk_res.map_err(|e| GraphError::Node {
            node: "agent".into(),
            source: anyhow::anyhow!(e.to_string()),
        })?;

        if !chunk.text.is_empty() {
            text.push_str(&chunk.text);
        }
        for delta in chunk.tool_call_chunks {
            let entry = tool_buffers.entry(delta.index).or_default();
            if let Some(id) = delta.id {
                entry.id = Some(id);
            }
            if let Some(name) = delta.name {
                entry.name = Some(name);
            }
            if let Some(args) = delta.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }

    let tool_calls: Vec<ToolCallRequest> = tool_buffers
        .into_values()
        .filter_map(|p| {
            let name = p.name?;
            let id = p.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let arguments = if p.arguments.is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(&p.arguments).unwrap_or(Value::String(p.arguments))
            };
            Some(ToolCallRequest { id, name, arguments })
        })
        .collect();

    Ok(if tool_calls.is_empty() {
        Message::ai(text)
    } else {
        Message::ai_with_tool_calls(text, tool_calls)
    })
}

#[derive(Default)]
struct PartialTool {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

// ---------------------------------------------------------------------------
// Value <-> Message conversions
// ---------------------------------------------------------------------------

fn json_to_message(v: &Value) -> Result<Message, String> {
    let obj = v.as_object().ok_or_else(|| "message is not an object".to_string())?;

    // Prefer `role`; fall back to `type` for the `{"type": "tool"}` shape
    // emitted by `ToolNode`.
    let role_str = obj
        .get("role")
        .and_then(|v| v.as_str())
        .or_else(|| obj.get("type").and_then(|v| v.as_str()))
        .unwrap_or("user");

    let role = match role_str {
        "system" => Role::System,
        "assistant" | "ai" => Role::Ai,
        "tool" => Role::Tool,
        _ => Role::Human,
    };

    let content_text = match obj.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };

    let tool_calls: Vec<ToolCallRequest> = obj
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let tc = tc.as_object()?;
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let name = tc.get("name").and_then(|v| v.as_str())?.to_string();
                    // Accept either `args` (graph-native shape) or `arguments`
                    // (OpenAI-native shape).
                    let arguments = tc
                        .get("args")
                        .or_else(|| tc.get("arguments"))
                        .cloned()
                        .unwrap_or(Value::Object(Default::default()));
                    Some(ToolCallRequest { id, name, arguments })
                })
                .collect()
        })
        .unwrap_or_default();

    let raw_tool_call_id = obj.get("tool_call_id").or_else(|| obj.get("id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let raw_name = obj.get("name").and_then(|v| v.as_str()).map(String::from);

    let is_tool = role == Role::Tool;

    Ok(Message {
        id: uuid::Uuid::new_v4().to_string(),
        role,
        content: if content_text.is_empty() {
            Vec::new()
        } else {
            vec![ContentBlock::Text { text: content_text }]
        },
        tool_calls,
        tool_call_id: if is_tool { raw_tool_call_id } else { None },
        name: if is_tool { raw_name } else { None },
    })
}

fn message_to_json(msg: &Message) -> Value {
    let role = msg.role.as_openai_str();
    let mut out = serde_json::Map::new();
    out.insert("role".into(), Value::String(role.into()));
    out.insert("content".into(), Value::String(msg.text()));
    if !msg.tool_calls.is_empty() {
        let calls: Vec<Value> = msg
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.arguments,
                })
            })
            .collect();
        out.insert("tool_calls".into(), Value::Array(calls));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_user_message() {
        let v = json!({"role": "user", "content": "hi"});
        let m = json_to_message(&v).unwrap();
        assert_eq!(m.role, Role::Human);
        assert_eq!(m.text(), "hi");
    }

    #[test]
    fn round_trip_ai_with_tool_calls() {
        let v = json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "calc", "args": {"a": 1}}]
        });
        let m = json_to_message(&v).unwrap();
        assert!(m.has_tool_calls());
        assert_eq!(m.tool_calls[0].name, "calc");
        let back = message_to_json(&m);
        assert_eq!(back["tool_calls"][0]["args"], json!({"a": 1}));
        assert_eq!(back["role"], "assistant");
    }

    #[test]
    fn tool_message_from_tool_node_shape() {
        let v = json!({
            "id": "c1",
            "type": "tool",
            "name": "calc",
            "content": {"sum": 3}
        });
        let m = json_to_message(&v).unwrap();
        assert_eq!(m.role, Role::Tool);
        assert_eq!(m.tool_call_id.as_deref(), Some("c1"));
        assert_eq!(m.name.as_deref(), Some("calc"));
    }

    #[test]
    fn message_to_json_omits_tool_calls_when_empty() {
        let v = message_to_json(&Message::ai("done"));
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "done");
        assert!(v.get("tool_calls").is_none());
    }
}
