//! GCP Vertex AI / Gemini provider.
//!
//! Uses the `generateContent` and `streamGenerateContent` endpoints on
//! `{location}-aiplatform.googleapis.com`. Auth flows through ADC via
//! `gcp_auth`, but static tokens can be injected for tests.

pub mod client;
pub mod stream;
pub mod types;

pub use client::VertexGeminiModel;
