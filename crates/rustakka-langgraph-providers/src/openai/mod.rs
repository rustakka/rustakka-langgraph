//! OpenAI chat completion provider.
//!
//! Also serves as the LiteLLM / vLLM provider — just set a custom `base_url`.

pub mod client;
pub mod stream;
pub mod types;

pub use client::OpenAiModel;

// Internal re-exports for sibling providers (Azure) that reuse the wire
// format and SSE parser.
#[cfg(feature = "azure")]
pub(crate) use stream::parse_sse_stream;
#[cfg(feature = "azure")]
pub(crate) use types::{ChatCompletionRequest, ChatCompletionResponse, WireMessage, WireTool};
