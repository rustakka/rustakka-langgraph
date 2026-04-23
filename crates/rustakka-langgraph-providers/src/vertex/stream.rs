//! Gemini `streamGenerateContent?alt=sse` parser.

use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};

use crate::error::ProviderError;
use crate::types::options::{GenerationChunk, ToolCallChunkDelta};

use super::types::{GeminiPart, GeminiResponse};

/// Convert a Gemini SSE stream into [`GenerationChunk`]s.
pub(crate) fn parse_gemini_sse(
    mut es: EventSource,
) -> BoxStream<'static, Result<GenerationChunk, ProviderError>> {
    let s = async_stream::stream! {
        let mut tool_call_index: usize = 0;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    let data = msg.data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    match serde_json::from_str::<GeminiResponse>(data) {
                        Ok(resp) => {
                            for candidate in resp.candidates {
                                let mut text = String::new();
                                let mut tc_chunks: Vec<ToolCallChunkDelta> = Vec::new();
                                for part in candidate.content.parts {
                                    match part {
                                        GeminiPart::Text { text: t } => text.push_str(&t),
                                        GeminiPart::FunctionCall { function_call } => {
                                            tc_chunks.push(ToolCallChunkDelta {
                                                index: tool_call_index,
                                                id: None,
                                                name: Some(function_call.name),
                                                arguments: Some(function_call.args.to_string()),
                                            });
                                            tool_call_index += 1;
                                        }
                                        _ => {}
                                    }
                                }
                                if !text.is_empty() || !tc_chunks.is_empty() {
                                    yield Ok(GenerationChunk {
                                        text,
                                        tool_call_chunks: tc_chunks,
                                        metadata: serde_json::json!({
                                            "finish_reason": candidate.finish_reason,
                                            "model": resp.model_version,
                                        }),
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            yield Err(ProviderError::Parse(format!(
                                "failed to parse Gemini SSE chunk: {e}: {data}"
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
