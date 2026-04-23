//! OpenAI wire-format request/response types.
//!
//! These are the serde structs matching the `/v1/chat/completions` JSON schema.
//! They are used internally by [`super::client::OpenAiModel`] and are not part
//! of the public API — consumers work with the unified [`Message`] type.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::message::{ContentBlock, Message, Role, ToolCallRequest};
use crate::types::options::ToolDefinition;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// OpenAI wire message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: WireFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: WireToolFunction,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(default)]
    pub usage: Option<Value>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionChoice {
    pub index: u32,
    pub message: WireMessage,
    pub finish_reason: Option<String>,
}

/// Streaming chunk (SSE `data: {...}` line).
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversions: internal Message <-> wire format
// ---------------------------------------------------------------------------

impl From<&Message> for WireMessage {
    fn from(msg: &Message) -> Self {
        let content = if msg.content.len() == 1 {
            // Single text block → plain string (most common case)
            msg.content.first().and_then(|b| match b {
                ContentBlock::Text { text } => Some(Value::String(text.clone())),
                _ => None,
            })
        } else if msg.content.is_empty() {
            None
        } else {
            // Multimodal → array of content parts
            let parts: Vec<Value> = msg
                .content
                .iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    ContentBlock::ImageUrl { url, detail } => {
                        let mut img = serde_json::json!({"url": url});
                        if let Some(d) = detail {
                            img["detail"] = Value::String(d.clone());
                        }
                        serde_json::json!({"type": "image_url", "image_url": img})
                    }
                })
                .collect();
            Some(Value::Array(parts))
        };

        let tool_calls = if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| WireToolCall {
                        id: tc.id.clone(),
                        call_type: "function".into(),
                        function: WireFunction {
                            name: tc.name.clone(),
                            arguments: tc.arguments.to_string(),
                        },
                    })
                    .collect(),
            )
        };

        WireMessage {
            role: msg.role.as_openai_str().to_string(),
            content,
            name: msg.name.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

impl WireMessage {
    /// Convert a wire response message into the unified `Message` type.
    pub(crate) fn into_message(self) -> Message {
        let role = Role::from_openai_str(&self.role).unwrap_or(Role::Ai);

        let content = match self.content {
            Some(Value::String(s)) => vec![ContentBlock::Text { text: s }],
            Some(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| {
                    let t = v.get("type")?.as_str()?;
                    match t {
                        "text" => {
                            let text = v.get("text")?.as_str()?.to_string();
                            Some(ContentBlock::Text { text })
                        }
                        _ => None,
                    }
                })
                .collect(),
            Some(Value::Null) | None => Vec::new(),
            _ => Vec::new(),
        };

        let tool_calls: Vec<ToolCallRequest> = self
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| {
                let args = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(Value::String(tc.function.arguments));
                ToolCallRequest {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: args,
                }
            })
            .collect();

        Message {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content,
            tool_calls,
            tool_call_id: self.tool_call_id,
            name: self.name,
        }
    }
}

impl From<&ToolDefinition> for WireTool {
    fn from(td: &ToolDefinition) -> Self {
        WireTool {
            tool_type: "function".into(),
            function: WireToolFunction {
                name: td.name.clone(),
                description: td.description.clone(),
                parameters: td.parameters.clone(),
            },
        }
    }
}
