//! Azure OpenAI provider.
//!
//! Reuses the OpenAI wire types verbatim; only the auth header (`api-key`)
//! and the URL template (`/openai/deployments/{deployment}/...`) differ.

pub mod client;

pub use client::AzureOpenAiModel;
