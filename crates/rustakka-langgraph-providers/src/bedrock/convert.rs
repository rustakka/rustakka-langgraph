//! Pure conversions between our unified [`Message`] type and the Bedrock
//! Converse SDK types. Extracted so they are directly unit-testable without
//! needing HTTP mocking.

use aws_sdk_bedrockruntime::types as br;
use aws_smithy_types::Document;
use serde_json::Value;

use crate::error::ProviderError;
use crate::types::message::{ContentBlock, Message, Role, ToolCallRequest};
use crate::types::options::ToolDefinition;

// ---------------------------------------------------------------------------
// serde_json::Value <-> aws_smithy_types::Document
// ---------------------------------------------------------------------------

pub(crate) fn json_to_document(v: &Value) -> Document {
    match v {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(u) = n.as_u64() {
                Document::Number(aws_smithy_types::Number::PosInt(u))
            } else {
                Document::Number(aws_smithy_types::Number::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        Value::String(s) => Document::String(s.clone()),
        Value::Array(arr) => Document::Array(arr.iter().map(json_to_document).collect()),
        Value::Object(obj) => {
            let mut m = std::collections::HashMap::with_capacity(obj.len());
            for (k, val) in obj {
                m.insert(k.clone(), json_to_document(val));
            }
            Document::Object(m)
        }
    }
}

pub(crate) fn document_to_json(d: &Document) -> Value {
    match d {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(v) => Value::from(*v),
            aws_smithy_types::Number::NegInt(v) => Value::from(*v),
            aws_smithy_types::Number::Float(v) => {
                serde_json::Number::from_f64(*v).map(Value::Number).unwrap_or(Value::Null)
            }
        },
        Document::String(s) => Value::String(s.clone()),
        Document::Array(arr) => Value::Array(arr.iter().map(document_to_json).collect()),
        Document::Object(obj) => {
            let mut m = serde_json::Map::with_capacity(obj.len());
            for (k, v) in obj {
                m.insert(k.clone(), document_to_json(v));
            }
            Value::Object(m)
        }
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Partition messages into (system blocks, turn messages).
///
/// Bedrock's Converse API takes the system prompt as a separate `system`
/// parameter rather than as a `system` role in the messages array.
pub(crate) fn messages_to_converse(
    messages: &[Message],
) -> Result<(Vec<br::SystemContentBlock>, Vec<br::Message>), ProviderError> {
    let mut system: Vec<br::SystemContentBlock> = Vec::new();
    let mut turns: Vec<br::Message> = Vec::new();

    for m in messages {
        match m.role {
            Role::System => {
                let text = m.text();
                if !text.is_empty() {
                    system.push(br::SystemContentBlock::Text(text));
                }
            }
            _ => turns.push(message_to_converse(m)?),
        }
    }

    Ok((system, turns))
}

fn message_to_converse(m: &Message) -> Result<br::Message, ProviderError> {
    let role = match m.role {
        Role::Ai => br::ConversationRole::Assistant,
        // Tool results ride on user turns per Bedrock's Converse spec.
        _ => br::ConversationRole::User,
    };

    let mut blocks: Vec<br::ContentBlock> = Vec::new();

    // Tool result message.
    if m.role == Role::Tool {
        let tool_use_id = m.tool_call_id.clone().unwrap_or_default();
        let content_value: Value = serde_json::from_str(&m.text())
            .unwrap_or(Value::String(m.text()));
        let block = br::ToolResultBlock::builder()
            .tool_use_id(tool_use_id)
            .content(br::ToolResultContentBlock::Json(json_to_document(&content_value)))
            .build()
            .map_err(|e| ProviderError::other(format!("bedrock tool result build: {e}")))?;
        blocks.push(br::ContentBlock::ToolResult(block));
        return Ok(br::Message::builder()
            .role(role)
            .set_content(Some(blocks))
            .build()
            .map_err(|e| ProviderError::other(format!("bedrock message build: {e}")))?);
    }

    // Text + image parts.
    for block in &m.content {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                blocks.push(br::ContentBlock::Text(text.clone()));
            }
            ContentBlock::Text { .. } => {}
            ContentBlock::ImageUrl { url, .. } => {
                // Converse expects base64 bytes, not URLs; surface URL as text
                // to avoid silent data loss. Callers wanting real images should
                // extend this conversion.
                blocks.push(br::ContentBlock::Text(format!("[image: {url}]")));
            }
        }
    }

    // Assistant tool-use requests.
    for tc in &m.tool_calls {
        let block = br::ToolUseBlock::builder()
            .tool_use_id(tc.id.clone())
            .name(tc.name.clone())
            .input(json_to_document(&tc.arguments))
            .build()
            .map_err(|e| ProviderError::other(format!("bedrock tool use build: {e}")))?;
        blocks.push(br::ContentBlock::ToolUse(block));
    }

    Ok(br::Message::builder()
        .role(role)
        .set_content(Some(blocks))
        .build()
        .map_err(|e| ProviderError::other(format!("bedrock message build: {e}")))?)
}

/// Transform a Converse output into a unified [`Message`].
pub(crate) fn converse_output_to_message(out: &br::Message) -> Message {
    let role = match out.role {
        br::ConversationRole::Assistant => Role::Ai,
        br::ConversationRole::User => Role::Human,
        _ => Role::Ai,
    };

    let mut content: Vec<ContentBlock> = Vec::new();
    let mut tool_calls: Vec<ToolCallRequest> = Vec::new();

    for block in &out.content {
        match block {
            br::ContentBlock::Text(t) => content.push(ContentBlock::Text { text: t.clone() }),
            br::ContentBlock::ToolUse(tu) => {
                tool_calls.push(ToolCallRequest {
                    id: tu.tool_use_id().to_string(),
                    name: tu.name().to_string(),
                    arguments: document_to_json(tu.input()),
                });
            }
            _ => {}
        }
    }

    Message {
        id: uuid::Uuid::new_v4().to_string(),
        role,
        content,
        tool_calls,
        tool_call_id: None,
        name: None,
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

pub(crate) fn tools_to_converse(
    tools: &[ToolDefinition],
) -> Result<Option<br::ToolConfiguration>, ProviderError> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut out: Vec<br::Tool> = Vec::with_capacity(tools.len());
    for td in tools {
        let spec = br::ToolSpecification::builder()
            .name(td.name.clone())
            .description(td.description.clone())
            .input_schema(br::ToolInputSchema::Json(json_to_document(&td.parameters)))
            .build()
            .map_err(|e| ProviderError::other(format!("bedrock tool spec build: {e}")))?;
        out.push(br::Tool::ToolSpec(spec));
    }
    Ok(Some(
        br::ToolConfiguration::builder()
            .set_tools(Some(out))
            .build()
            .map_err(|e| ProviderError::other(format!("bedrock tool config build: {e}")))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_to_document_roundtrip() {
        let v = json!({"a": 1, "b": [true, "str"], "c": null, "d": 3.14});
        let d = json_to_document(&v);
        let back = document_to_json(&d);
        assert_eq!(back, v);
    }

    #[test]
    fn system_message_separated() {
        let (sys, turns) = messages_to_converse(&[
            Message::system("helpful"),
            Message::human("hi"),
        ])
        .unwrap();
        assert_eq!(sys.len(), 1);
        assert_eq!(turns.len(), 1);
    }

    #[test]
    fn tool_result_block_from_tool_message() {
        let tool = Message::tool("call_1", "add", r#"{"sum": 3}"#);
        let (_, turns) = messages_to_converse(&[tool]).unwrap();
        assert_eq!(turns.len(), 1);
        let blocks = &turns[0].content;
        let has_tool_result = blocks
            .iter()
            .any(|b| matches!(b, br::ContentBlock::ToolResult(_)));
        assert!(has_tool_result);
    }

    #[test]
    fn ai_tool_call_becomes_tool_use_block() {
        let ai = Message::ai_with_tool_calls(
            "calling",
            vec![ToolCallRequest {
                id: "t1".into(),
                name: "calc".into(),
                arguments: json!({"a": 1}),
            }],
        );
        let (_, turns) = messages_to_converse(&[ai]).unwrap();
        assert_eq!(turns.len(), 1);
        let has_tool_use = turns[0]
            .content
            .iter()
            .any(|b| matches!(b, br::ContentBlock::ToolUse(_)));
        assert!(has_tool_use);
    }

    #[test]
    fn converse_output_extracts_tool_calls() {
        let tu = br::ToolUseBlock::builder()
            .tool_use_id("tu1")
            .name("calc")
            .input(json_to_document(&json!({"a": 1, "b": 2})))
            .build()
            .unwrap();
        let out = br::Message::builder()
            .role(br::ConversationRole::Assistant)
            .set_content(Some(vec![
                br::ContentBlock::Text("calling".into()),
                br::ContentBlock::ToolUse(tu),
            ]))
            .build()
            .unwrap();
        let msg = converse_output_to_message(&out);
        assert_eq!(msg.text(), "calling");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn tools_to_converse_builds_configuration() {
        let tds = vec![ToolDefinition {
            name: "calc".into(),
            description: "add".into(),
            parameters: json!({"type": "object"}),
        }];
        let cfg = tools_to_converse(&tds).unwrap().unwrap();
        assert_eq!(cfg.tools().len(), 1);
    }
}
