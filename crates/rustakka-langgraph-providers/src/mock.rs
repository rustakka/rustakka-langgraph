//! Deterministic mock provider for testing graph flows without network I/O.
//!
//! `MockChatModel` returns canned responses in FIFO order. For streaming it
//! splits the canned text on whitespace and yields one chunk per word with
//! an optional simulated latency between chunks.

use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use crate::error::ProviderError;
use crate::traits::ChatModel;
use crate::types::message::Message;
use crate::types::options::{CallOptions, GenerationChunk};

/// A test-double [`ChatModel`] that returns pre-programmed responses.
///
/// # Examples
///
/// ```rust,ignore
/// let mock = MockChatModel::new(vec![
///     Message::ai("The answer is 42."),
/// ]);
/// let reply = mock.invoke(&[Message::human("Hi")], &CallOptions::default()).await?;
/// assert_eq!(reply.text(), "The answer is 42.");
/// ```
pub struct MockChatModel {
    /// FIFO queue of responses. Each `invoke` / `stream` call pops the front.
    responses: Mutex<Vec<Message>>,
    /// Optional per-chunk latency injected during `stream`.
    pub chunk_latency: Option<Duration>,
    /// Model name returned by [`ChatModel::model_name`].
    name: String,
}

impl std::fmt::Debug for MockChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let remaining = self.responses.lock().map(|q| q.len()).unwrap_or(0);
        f.debug_struct("MockChatModel")
            .field("name", &self.name)
            .field("remaining_responses", &remaining)
            .field("chunk_latency", &self.chunk_latency)
            .finish()
    }
}

impl MockChatModel {
    /// Create a mock with a queue of canned responses returned in order.
    pub fn new(responses: Vec<Message>) -> Self {
        Self {
            responses: Mutex::new(responses),
            chunk_latency: None,
            name: "mock".into(),
        }
    }

    /// Set the model name (default: `"mock"`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Inject latency between streamed chunks.
    pub fn with_chunk_latency(mut self, d: Duration) -> Self {
        self.chunk_latency = Some(d);
        self
    }

    fn pop_response(&self) -> Result<Message, ProviderError> {
        let mut q = self
            .responses
            .lock()
            .map_err(|e| ProviderError::Other(format!("lock poisoned: {e}")))?;
        if q.is_empty() {
            Err(ProviderError::Other(
                "MockChatModel: no more canned responses".into(),
            ))
        } else {
            Ok(q.remove(0))
        }
    }
}

#[async_trait]
impl ChatModel for MockChatModel {
    async fn invoke(
        &self,
        _messages: &[Message],
        _options: &CallOptions,
    ) -> Result<Message, ProviderError> {
        self.pop_response()
    }

    async fn stream(
        &self,
        _messages: &[Message],
        _options: &CallOptions,
    ) -> Result<BoxStream<'_, Result<GenerationChunk, ProviderError>>, ProviderError> {
        let msg = self.pop_response()?;

        // If the message has tool calls, yield them in one chunk then finish
        if msg.has_tool_calls() {
            let tool_chunks: Vec<_> = msg
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| crate::types::options::ToolCallChunkDelta {
                    index: i,
                    id: Some(tc.id.clone()),
                    name: Some(tc.name.clone()),
                    arguments: Some(tc.arguments.to_string()),
                })
                .collect();
            let chunks = vec![
                Ok(GenerationChunk {
                    text: msg.text(),
                    tool_call_chunks: tool_chunks,
                    ..Default::default()
                }),
            ];
            return Ok(Box::pin(stream::iter(chunks)));
        }

        // Split text on whitespace → one chunk per word
        let words: Vec<String> = msg
            .text()
            .split_whitespace()
            .map(String::from)
            .collect();

        let latency = self.chunk_latency;
        let s = stream::iter(words).then(move |word| async move {
            if let Some(d) = latency {
                tokio::time::sleep(d).await;
            }
            Ok(GenerationChunk::text(word))
        });

        Ok(Box::pin(s))
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::ToolCallRequest;
    use futures::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn invoke_returns_fifo() {
        let mock = MockChatModel::new(vec![
            Message::ai("first"),
            Message::ai("second"),
        ]);
        let opts = CallOptions::default();

        let r1 = mock.invoke(&[Message::human("a")], &opts).await.unwrap();
        assert_eq!(r1.text(), "first");

        let r2 = mock.invoke(&[Message::human("b")], &opts).await.unwrap();
        assert_eq!(r2.text(), "second");

        // exhausted
        assert!(mock.invoke(&[Message::human("c")], &opts).await.is_err());
    }

    #[tokio::test]
    async fn stream_splits_on_whitespace() {
        let mock = MockChatModel::new(vec![
            Message::ai("hello world foo"),
        ]);
        let opts = CallOptions::default();
        let mut s = mock.stream(&[Message::human("hi")], &opts).await.unwrap();

        let mut words = Vec::new();
        while let Some(chunk) = s.next().await {
            words.push(chunk.unwrap().text);
        }
        assert_eq!(words, vec!["hello", "world", "foo"]);
    }

    #[tokio::test]
    async fn stream_with_tool_calls() {
        let mock = MockChatModel::new(vec![
            Message::ai_with_tool_calls(
                "calling",
                vec![ToolCallRequest {
                    id: "t1".into(),
                    name: "calc".into(),
                    arguments: json!({"a": 1}),
                }],
            ),
        ]);
        let opts = CallOptions::default();
        let mut s = mock.stream(&[Message::human("hi")], &opts).await.unwrap();

        let chunk = s.next().await.unwrap().unwrap();
        assert_eq!(chunk.tool_call_chunks.len(), 1);
        assert_eq!(chunk.tool_call_chunks[0].name.as_deref(), Some("calc"));
    }
}
