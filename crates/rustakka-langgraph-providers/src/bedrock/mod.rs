//! AWS Bedrock provider.
//!
//! Uses the Bedrock **Converse** / **ConverseStream** API, which normalizes
//! Claude, Llama, Mistral, and Titan into a single message shape. Legacy
//! `InvokeModel` endpoints with family-specific payloads are still reachable
//! via a user-supplied [`BedrockAdapter`](adapters::BedrockAdapter), but that
//! path is off the main road.

pub mod adapters;
pub mod client;
pub mod convert;

pub use client::BedrockModel;
