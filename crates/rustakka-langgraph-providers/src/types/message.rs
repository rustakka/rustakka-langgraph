//! Unified message and content types for all LLM providers.
//!
//! These mirror LangChain's `BaseMessage` hierarchy while staying idiomatic
//! Rust. Every provider translates between these types and its wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// Conversation role, mapped to provider-specific strings on the wire.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System-level instructions.
    System,
    /// End-user input (wire: `"user"`).
    Human,
    /// Model output (wire: `"assistant"`).
    Ai,
    /// Tool / function result.
    Tool,
}

impl Role {
    /// Wire-format string used by the OpenAI family of APIs.
    pub fn as_openai_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Human => "user",
            Role::Ai => "assistant",
            Role::Tool => "tool",
        }
    }

    /// Parse from the OpenAI wire-format string.
    pub fn from_openai_str(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::Human),
            "assistant" => Some(Role::Ai),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

/// A single block of content within a message. Simple text messages contain
/// one `Text` block; multimodal prompts may contain several blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text { text: String },
    /// Remote image referenced by URL.
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

impl ContentBlock {
    /// Convenience: extract text if this is a `Text` block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool call
// ---------------------------------------------------------------------------

/// A request from the model to invoke a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRequest {
    /// Unique identifier for this call (used to correlate `ToolMessage`).
    pub id: String,
    /// Tool / function name.
    pub name: String,
    /// Tool arguments as a JSON value (usually an object).
    #[serde(default)]
    pub arguments: Value,
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// Unified message type consumed and produced by all providers.
///
/// Design mirrors LangChain's `BaseMessage` but is a single struct with
/// role discrimination — avoids the combinatorial explosion of separate
/// types while preserving strong typing via the `Role` enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message identifier (auto-generated if not supplied).
    pub id: String,
    /// Conversation role.
    pub role: Role,
    /// Message body. Simple text messages use a single [`ContentBlock::Text`];
    /// multimodal prompts use multiple blocks.
    pub content: Vec<ContentBlock>,
    /// Tool-call requests (non-empty only on `Role::Ai` messages).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRequest>,
    /// ID of the tool call this message answers (only on `Role::Tool`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name (only on `Role::Tool` messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    // ---- Convenience constructors ----

    /// Create a system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a human (user) message.
    pub fn human(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Human,
            content: vec![ContentBlock::Text { text: text.into() }],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an AI (assistant) message with text content.
    pub fn ai(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Ai,
            content: vec![ContentBlock::Text { text: text.into() }],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an AI message that requests tool invocations.
    pub fn ai_with_tool_calls(
        text: impl Into<String>,
        tool_calls: Vec<ToolCallRequest>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Ai,
            content: vec![ContentBlock::Text { text: text.into() }],
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a tool-result message answering a specific tool call.
    pub fn tool(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Tool,
            content: vec![ContentBlock::Text { text: content.into() }],
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            name: Some(tool_name.into()),
        }
    }

    /// Create a human message with mixed content blocks (text + images).
    pub fn human_multimodal(blocks: Vec<ContentBlock>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Human,
            content: blocks,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    // ---- Accessors ----

    /// Concatenate all text content blocks into a single string.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Does this message contain tool-call requests?
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn system_message_roundtrip() {
        let msg = Message::system("You are helpful.");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "system");
        let back: Message = serde_json::from_value(json).unwrap();
        assert_eq!(back.role, Role::System);
        assert_eq!(back.text(), "You are helpful.");
    }

    #[test]
    fn ai_with_tool_calls_roundtrip() {
        let msg = Message::ai_with_tool_calls(
            "calling calc",
            vec![ToolCallRequest {
                id: "t1".into(),
                name: "calc".into(),
                arguments: json!({"a": 1, "b": 2}),
            }],
        );
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["tool_calls"][0]["name"], "calc");
        let back: Message = serde_json::from_value(json).unwrap();
        assert!(back.has_tool_calls());
        assert_eq!(back.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn tool_message_roundtrip() {
        let msg = Message::tool("t1", "calc", r#"{"sum": 3}"#);
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "t1");
        assert_eq!(json["name"], "calc");
        let back: Message = serde_json::from_value(json).unwrap();
        assert_eq!(back.tool_call_id.as_deref(), Some("t1"));
    }

    #[test]
    fn role_openai_wire_format() {
        assert_eq!(Role::Human.as_openai_str(), "user");
        assert_eq!(Role::Ai.as_openai_str(), "assistant");
        assert_eq!(Role::from_openai_str("user"), Some(Role::Human));
        assert_eq!(Role::from_openai_str("assistant"), Some(Role::Ai));
        assert_eq!(Role::from_openai_str("bogus"), None);
    }

    #[test]
    fn multimodal_message() {
        let msg = Message::human_multimodal(vec![
            ContentBlock::Text { text: "What's in this image?".into() },
            ContentBlock::ImageUrl {
                url: "https://example.com/img.png".into(),
                detail: Some("high".into()),
            },
        ]);
        assert_eq!(msg.content.len(), 2);
        assert_eq!(msg.text(), "What's in this image?");
    }
}
