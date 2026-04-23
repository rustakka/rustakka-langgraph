//! OpenAI SSE stream parser.
//!
//! Parses Server-Sent Events from the `/v1/chat/completions` streaming
//! endpoint into [`GenerationChunk`]s.

use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};

use crate::error::ProviderError;
use crate::types::options::{GenerationChunk, ToolCallChunkDelta};

use super::types::ChatCompletionChunk;

/// Convert a `reqwest_eventsource::EventSource` into a stream of
/// [`GenerationChunk`]s.
pub(crate) fn parse_sse_stream(
    mut es: EventSource,
) -> BoxStream<'static, Result<GenerationChunk, ProviderError>> {
    let s = async_stream::stream! {
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => { /* connection opened */ }
                Ok(Event::Message(msg)) => {
                    let data = msg.data.trim();
                    if data == "[DONE]" {
                        break;
                    }
                    match serde_json::from_str::<ChatCompletionChunk>(data) {
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                let text = choice.delta.content.clone().unwrap_or_default();
                                let tool_call_chunks: Vec<ToolCallChunkDelta> = choice
                                    .delta
                                    .tool_calls
                                    .as_ref()
                                    .map(|tcs| {
                                        tcs.iter()
                                            .map(|tc| ToolCallChunkDelta {
                                                index: tc.index,
                                                id: tc.id.clone(),
                                                name: tc.function.as_ref().and_then(|f| f.name.clone()),
                                                arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();

                                if !text.is_empty() || !tool_call_chunks.is_empty() {
                                    yield Ok(GenerationChunk {
                                        text,
                                        tool_call_chunks,
                                        metadata: serde_json::json!({
                                            "model": chunk.model,
                                            "finish_reason": choice.finish_reason,
                                        }),
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            yield Err(ProviderError::Parse(format!(
                                "failed to parse SSE chunk: {e}: {data}"
                            )));
                        }
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => {
                    yield Err(ProviderError::Stream(e.to_string()));
                    break;
                }
            }
        }
        es.close();
    };

    Box::pin(s)
}
