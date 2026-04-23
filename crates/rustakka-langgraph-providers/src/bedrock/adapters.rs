//! Legacy `InvokeModel` payload adapters.
//!
//! The Converse API (used by [`BedrockModel`](super::BedrockModel)) already
//! normalizes Claude / Llama / Titan payloads, so most callers will never need
//! these. They exist for niche cases where a caller must hit the raw
//! `InvokeModel` endpoint with a family-specific payload shape.

use serde_json::Value;

use crate::error::ProviderError;
use crate::types::message::Message;
use crate::types::options::CallOptions;

/// Transforms a generic request into the JSON body expected by a specific
/// Bedrock model family's `InvokeModel` endpoint.
pub trait BedrockAdapter: Send + Sync + std::fmt::Debug {
    /// Build the raw JSON body for `InvokeModel`.
    fn build_body(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Value, ProviderError>;

    /// Extract the assistant text from the raw JSON response.
    fn parse_response(&self, body: &Value) -> Result<Message, ProviderError>;
}

/// Adapter for the Anthropic Claude 3 Messages API format on Bedrock.
///
/// This is mostly useful if a caller explicitly wants to bypass Converse.
#[derive(Debug, Default)]
pub struct Claude3Adapter;

impl BedrockAdapter for Claude3Adapter {
    fn build_body(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Value, ProviderError> {
        let mut system = String::new();
        let mut msgs: Vec<Value> = Vec::new();
        for m in messages {
            match m.role {
                crate::types::message::Role::System => {
                    if !system.is_empty() {
                        system.push('\n');
                    }
                    system.push_str(&m.text());
                }
                crate::types::message::Role::Human => msgs.push(serde_json::json!({
                    "role": "user",
                    "content": m.text(),
                })),
                crate::types::message::Role::Ai => msgs.push(serde_json::json!({
                    "role": "assistant",
                    "content": m.text(),
                })),
                crate::types::message::Role::Tool => msgs.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.text(),
                    }]
                })),
            }
        }

        let mut body = serde_json::Map::new();
        body.insert("anthropic_version".into(), Value::String("bedrock-2023-05-31".into()));
        body.insert(
            "max_tokens".into(),
            Value::from(options.max_tokens.unwrap_or(1024)),
        );
        body.insert("messages".into(), Value::Array(msgs));
        if !system.is_empty() {
            body.insert("system".into(), Value::String(system));
        }
        if let Some(t) = options.temperature {
            body.insert("temperature".into(), Value::from(t));
        }
        Ok(Value::Object(body))
    }

    fn parse_response(&self, body: &Value) -> Result<Message, ProviderError> {
        // Claude returns { content: [{ type: "text", text: "..." }], ... }
        let text = body
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                arr.iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("")
                    .into()
            })
            .unwrap_or_default();
        Ok(Message::ai(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude3_builds_system_and_messages() {
        let adapter = Claude3Adapter;
        let body = adapter
            .build_body(
                &[
                    Message::system("be helpful"),
                    Message::human("hi"),
                    Message::ai("hello"),
                ],
                &CallOptions {
                    temperature: Some(0.1),
                    max_tokens: Some(256),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(body["system"], "be helpful");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["max_tokens"], 256);
    }

    #[test]
    fn claude3_parses_response() {
        let adapter = Claude3Adapter;
        let body = serde_json::json!({
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"}
            ]
        });
        let msg = adapter.parse_response(&body).unwrap();
        assert_eq!(msg.text(), "Hello world");
    }
}
