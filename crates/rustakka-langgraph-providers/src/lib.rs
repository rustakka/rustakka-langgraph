//! # rustakka-langgraph-providers
//!
//! LLM provider integrations for rustakka-langgraph. Each provider implements
//! the [`ChatModel`] trait, giving graph nodes a uniform interface for
//! invoke/stream calls across OpenAI, Ollama, AWS Bedrock, Azure, and GCP.
//!
//! ## Feature Flags
//!
//! - `openai` *(default)* — OpenAI + LiteLLM/vLLM via configurable base URL.
//! - `ollama` — Local Ollama server with NDJSON streaming.
//! - `bedrock` — AWS Bedrock (Claude, Llama, Titan).
//! - `azure` — Azure OpenAI deployments.
//! - `vertex` — GCP Vertex AI / Gemini.
//! - `mock` — Deterministic mock for testing.
//! - `full` — Everything.

#![forbid(unsafe_code)]

pub mod error;
pub mod traits;
pub mod types;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "vertex")]
pub mod vertex;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "mock")]
pub mod mock;

pub mod prelude {
    //! Common imports for provider consumers.
    pub use crate::error::ProviderError;
    pub use crate::traits::ChatModel;
    pub use crate::types::message::{ContentBlock, Message, Role, ToolCallRequest};
    pub use crate::types::options::{CallOptions, GenerationChunk, ToolCallChunkDelta, ToolDefinition};

    #[cfg(feature = "mock")]
    pub use crate::mock::MockChatModel;

    #[cfg(feature = "openai")]
    pub use crate::openai::OpenAiModel;

    #[cfg(feature = "ollama")]
    pub use crate::ollama::OllamaModel;

    #[cfg(feature = "vertex")]
    pub use crate::vertex::{client::VertexAuth, VertexGeminiModel};

    #[cfg(feature = "bedrock")]
    pub use crate::bedrock::BedrockModel;

    #[cfg(feature = "azure")]
    pub use crate::azure::AzureOpenAiModel;
}
