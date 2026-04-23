//! Per-request options and streaming chunk types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// JSON-Schema-based definition of a tool that the model can invoke.
///
/// Passed to the provider inside [`CallOptions::tools`]; providers map this
/// to their wire format (e.g., OpenAI `tools[].function`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool / function name (must match the registry key in `ToolNode`).
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema describing the expected arguments object.
    pub parameters: Value,
}

// ---------------------------------------------------------------------------
// Call options
// ---------------------------------------------------------------------------

/// Per-request options forwarded to the provider alongside the messages.
///
/// Fields default to `None` / empty, meaning "use the provider's default".
#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    /// Sampling temperature (0.0 – 2.0 on most providers).
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Stop sequences — generation halts when any of these is produced.
    pub stop: Vec<String>,
    /// Tool definitions the model is allowed to invoke.
    pub tools: Vec<ToolDefinition>,
    /// Tool-choice constraint: `"auto"`, `"required"`, `"none"`, or a
    /// specific tool name.
    pub tool_choice: Option<String>,
    /// Response format hint (e.g., `{"type": "json_object"}`).
    pub response_format: Option<Value>,
    /// Arbitrary provider-specific overrides. These are merged into the
    /// request body at the provider level.
    pub extra: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Generation chunks (streaming)
// ---------------------------------------------------------------------------

/// Incremental delta for a tool call being streamed.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCallChunkDelta {
    /// Index within the `tool_calls` array (for multi-tool parallel calls).
    pub index: usize,
    /// Tool-call ID (present only in the first chunk for this index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool name (present only in the first chunk for this index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Partial JSON string of the arguments being streamed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A single chunk yielded by [`ChatModel::stream`].
///
/// The consumer assembles a full [`Message`] by concatenating `text` fields
/// and merging `tool_call_chunks` across all chunks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationChunk {
    /// Incremental text content (may be empty on tool-call-only chunks).
    #[serde(default)]
    pub text: String,
    /// Incremental tool-call deltas.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_call_chunks: Vec<ToolCallChunkDelta>,
    /// Provider-specific metadata (model name, usage stats, finish_reason, …).
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl GenerationChunk {
    /// Create a text-only chunk.
    pub fn text(t: impl Into<String>) -> Self {
        Self { text: t.into(), ..Default::default() }
    }

    /// Create a chunk carrying only tool-call deltas.
    pub fn tool_calls(chunks: Vec<ToolCallChunkDelta>) -> Self {
        Self { tool_call_chunks: chunks, ..Default::default() }
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
    fn tool_definition_roundtrip() {
        let td = ToolDefinition {
            name: "calc".into(),
            description: "Add two numbers".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }),
        };
        let v = serde_json::to_value(&td).unwrap();
        let back: ToolDefinition = serde_json::from_value(v).unwrap();
        assert_eq!(back.name, "calc");
    }

    #[test]
    fn generation_chunk_defaults() {
        let c = GenerationChunk::default();
        assert!(c.text.is_empty());
        assert!(c.tool_call_chunks.is_empty());
        assert!(c.metadata.is_null());
    }

    #[test]
    fn generation_chunk_text_constructor() {
        let c = GenerationChunk::text("hello");
        assert_eq!(c.text, "hello");
    }
}
