//! Vertex / Gemini wire-format request and response types.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::message::{ContentBlock, Message, Role, ToolCallRequest};
use crate::types::options::ToolDefinition;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(crate) struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiContent {
    /// `"user"` or `"model"`. Not present on system instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiResponse {
    #[serde(default)]
    pub candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    pub usage_metadata: Option<Value>,
    #[serde(default, rename = "modelVersion")]
    pub model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiCandidate {
    pub content: GeminiContent,
    #[serde(default, rename = "finishReason")]
    pub finish_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Split a slice of unified messages into (system_instruction, contents).
///
/// Gemini treats the system prompt as a top-level field, so system messages
/// must be pulled out rather than embedded into `contents`.
pub(crate) fn messages_to_request_parts(
    messages: &[Message],
) -> (Option<GeminiContent>, Vec<GeminiContent>) {
    let mut system_text = String::new();
    let mut contents: Vec<GeminiContent> = Vec::new();

    for m in messages {
        match m.role {
            Role::System => {
                if !system_text.is_empty() {
                    system_text.push('\n');
                }
                system_text.push_str(&m.text());
            }
            _ => contents.push(message_to_content(m)),
        }
    }

    let system = if system_text.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: None,
            parts: vec![GeminiPart::Text { text: system_text }],
        })
    };
    (system, contents)
}

fn message_to_content(m: &Message) -> GeminiContent {
    let role = match m.role {
        Role::Ai => Some("model".to_string()),
        Role::Tool => Some("user".to_string()), // function responses ride on user turns
        _ => Some("user".to_string()),
    };

    let mut parts: Vec<GeminiPart> = Vec::new();
    for block in &m.content {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                parts.push(GeminiPart::Text { text: text.clone() });
            }
            ContentBlock::Text { .. } => {}
            ContentBlock::ImageUrl { url, .. } => {
                // best-effort: gemini expects base64-encoded inline data; if a
                // caller passes a URL, surface it as text so the model sees
                // something rather than silently dropping it.
                parts.push(GeminiPart::Text {
                    text: format!("[image: {url}]"),
                });
            }
        }
    }

    // Assistant tool-call requests.
    for tc in &m.tool_calls {
        parts.push(GeminiPart::FunctionCall {
            function_call: GeminiFunctionCall {
                name: tc.name.clone(),
                args: tc.arguments.clone(),
            },
        });
    }

    // Tool result messages.
    if m.role == Role::Tool {
        let name = m.name.clone().unwrap_or_default();
        let response = serde_json::from_str::<Value>(&m.text())
            .unwrap_or(Value::String(m.text()));
        parts = vec![GeminiPart::FunctionResponse {
            function_response: GeminiFunctionResponse { name, response },
        }];
    }

    GeminiContent { role, parts }
}

impl From<&ToolDefinition> for GeminiFunctionDeclaration {
    fn from(td: &ToolDefinition) -> Self {
        GeminiFunctionDeclaration {
            name: td.name.clone(),
            description: td.description.clone(),
            parameters: td.parameters.clone(),
        }
    }
}

/// Assemble a unified `Message` from the first candidate in a Gemini response.
pub(crate) fn response_to_message(resp: GeminiResponse) -> Message {
    let candidate = match resp.candidates.into_iter().next() {
        Some(c) => c,
        None => {
            return Message {
                id: uuid::Uuid::new_v4().to_string(),
                role: Role::Ai,
                content: Vec::new(),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            };
        }
    };

    let mut content_blocks: Vec<ContentBlock> = Vec::new();
    let mut tool_calls: Vec<ToolCallRequest> = Vec::new();

    for part in candidate.content.parts {
        match part {
            GeminiPart::Text { text } => {
                content_blocks.push(ContentBlock::Text { text });
            }
            GeminiPart::FunctionCall { function_call } => {
                tool_calls.push(ToolCallRequest {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: function_call.name,
                    arguments: function_call.args,
                });
            }
            // Ignore inline/function-response parts on outputs.
            _ => {}
        }
    }

    Message {
        id: uuid::Uuid::new_v4().to_string(),
        role: Role::Ai,
        content: content_blocks,
        tool_calls,
        tool_call_id: None,
        name: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn system_message_goes_to_system_instruction() {
        let (sys, contents) = messages_to_request_parts(&[
            Message::system("you are helpful"),
            Message::human("hi"),
        ]);
        let sys = sys.unwrap();
        match &sys.parts[0] {
            GeminiPart::Text { text } => assert_eq!(text, "you are helpful"),
            _ => panic!(),
        }
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
    }

    #[test]
    fn tool_message_becomes_function_response() {
        let tool = Message::tool("call_1", "add", r#"{"sum": 3}"#);
        let (_, contents) = messages_to_request_parts(&[tool]);
        assert_eq!(contents.len(), 1);
        match &contents[0].parts[0] {
            GeminiPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "add");
                assert_eq!(function_response.response, json!({"sum": 3}));
            }
            other => panic!("expected function response, got {other:?}"),
        }
    }

    #[test]
    fn response_to_message_extracts_tool_calls() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![
                        GeminiPart::Text { text: "calling".into() },
                        GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: "add".into(),
                                args: json!({"a": 1, "b": 2}),
                            },
                        },
                    ],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
            model_version: Some("gemini-1.5-flash".into()),
        };
        let msg = response_to_message(resp);
        assert_eq!(msg.text(), "calling");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
    }
}
