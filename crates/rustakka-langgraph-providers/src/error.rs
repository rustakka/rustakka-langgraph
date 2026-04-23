//! Provider error types.
//!
//! [`ProviderError`] is the top-level error returned by all [`ChatModel`]
//! implementations. It is deliberately kept separate from
//! [`rustakka_langgraph_core::errors::GraphError`] — callers that embed a
//! provider inside a graph node should map `ProviderError` → `GraphError::Node`.

use thiserror::Error;

/// Errors originating from LLM provider calls.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// The underlying HTTP transport failed (connection refused, timeout, …).
    #[error("HTTP request failed: {0}")]
    Http(String),

    /// The provider returned a non-2xx status code.
    #[error("provider returned error status {status}: {body}")]
    ApiError {
        status: u16,
        body: String,
    },

    /// Response JSON could not be deserialized into the expected schema.
    #[error("failed to parse provider response: {0}")]
    Parse(String),

    /// API key / token / credential problem.
    #[error("authentication error: {0}")]
    Auth(String),

    /// Provider signaled 429 / rate-limit.
    #[error("rate limited — retry after {retry_after_ms:?}ms")]
    RateLimited {
        retry_after_ms: Option<u64>,
    },

    /// Error while reading a streaming response (SSE or NDJSON).
    #[error("streaming error: {0}")]
    Stream(String),

    /// Catch-all for provider-specific issues.
    #[error("{0}")]
    Other(String),
}

impl ProviderError {
    pub fn other(msg: impl Into<String>) -> Self {
        ProviderError::Other(msg.into())
    }
}

#[cfg(any(feature = "openai", feature = "ollama", feature = "vertex", feature = "azure"))]
impl From<reqwest::Error> for ProviderError {
    fn from(e: reqwest::Error) -> Self {
        ProviderError::Http(e.to_string())
    }
}
