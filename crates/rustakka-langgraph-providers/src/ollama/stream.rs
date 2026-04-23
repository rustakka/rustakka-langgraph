//! Ollama NDJSON stream parser.

use futures::stream::BoxStream;
use futures::StreamExt;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;

use crate::error::ProviderError;
use crate::types::options::{GenerationChunk, ToolCallChunkDelta};
use super::client::OllamaChatResponse;

/// Parse an NDJSON byte stream from Ollama into [`GenerationChunk`]s.
pub(crate) fn parse_ndjson_stream(
    resp: reqwest::Response,
) -> BoxStream<'static, Result<GenerationChunk, ProviderError>> {
    let byte_stream = resp.bytes_stream().map(|r| {
        r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    });
    let reader = StreamReader::new(byte_stream);
    let lines = tokio::io::BufReader::new(reader).lines();

    let s = async_stream::stream! {
        tokio::pin!(lines);
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() { continue; }
            match serde_json::from_str::<OllamaChatResponse>(line.trim()) {
                Ok(resp) => {
                    let text = resp.message.as_ref()
                        .and_then(|m| m.content.clone())
                        .unwrap_or_default();
                    let tc_chunks: Vec<ToolCallChunkDelta> = resp.message.as_ref()
                        .and_then(|m| m.tool_calls.as_ref())
                        .map(|tcs| tcs.iter().enumerate().map(|(i, tc)| ToolCallChunkDelta {
                            index: i,
                            id: None,
                            name: tc.function.as_ref().map(|f| f.name.clone()),
                            arguments: tc.function.as_ref().map(|f| serde_json::to_string(&f.arguments).unwrap_or_default()),
                        }).collect())
                        .unwrap_or_default();
                    if !text.is_empty() || !tc_chunks.is_empty() {
                        yield Ok(GenerationChunk {
                            text,
                            tool_call_chunks: tc_chunks,
                            metadata: serde_json::json!({"model": resp.model, "done": resp.done}),
                        });
                    }
                    if resp.done { break; }
                }
                Err(e) => {
                    yield Err(ProviderError::Parse(format!("NDJSON parse error: {e}")));
                }
            }
        }
    };
    Box::pin(s)
}
