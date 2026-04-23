//! The [`ChatModel`] trait — the core abstraction every LLM provider implements.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use rustakka_streams::Source;
use tokio::sync::mpsc;

use crate::error::ProviderError;
use crate::types::message::Message;
use crate::types::options::{CallOptions, GenerationChunk};

/// Core interface for LLM chat completion providers.
///
/// Every provider (OpenAI, Ollama, Bedrock, …) implements this trait.
/// `Send + Sync` is required so the model can be shared across rustakka actor
/// boundaries — actors are `Send + 'static`, and a model typically lives
/// behind an `Arc<dyn ChatModel>`.
///
/// # Examples
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use rustakka_langgraph_providers::prelude::*;
///
/// async fn run(model: Arc<dyn ChatModel>) -> Result<(), ProviderError> {
///     let msgs = vec![
///         Message::system("You are helpful."),
///         Message::human("What is 2+2?"),
///     ];
///     let reply = model.invoke(&msgs, &CallOptions::default()).await?;
///     println!("{}", reply.text());
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait ChatModel: Send + Sync + std::fmt::Debug {
    /// One-shot invocation: send messages, receive a single AI response.
    ///
    /// The returned [`Message`] will have `role == Role::Ai` and may contain
    /// `tool_calls` if the model decides to invoke tools.
    async fn invoke(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Message, ProviderError>;

    /// Streaming invocation returning an async [`Stream`] of [`GenerationChunk`]s.
    ///
    /// The caller is responsible for assembling chunks into a final
    /// [`Message`]. Intermediate chunks are also suitable for forwarding to
    /// a [`StreamBus`] for real-time UI rendering.
    ///
    /// [`Stream`]: futures::Stream
    /// [`StreamBus`]: rustakka_langgraph_core::stream::StreamBus
    async fn stream(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<BoxStream<'_, Result<GenerationChunk, ProviderError>>, ProviderError>;

    /// Human-readable model identifier (e.g. `"gpt-4o"`, `"llama3:8b"`).
    fn model_name(&self) -> &str;
}

/// Additive rustakka-streams entry point. Wraps [`ChatModel::stream`] in
/// a `Source<Result<GenerationChunk, ProviderError>>` that owns its own
/// `Arc<dyn ChatModel>` — the caller can compose it with
/// [`rustakka_streams`] operators (`map`, `runWith`, overflow, kill
/// switch) without worrying about the original `&self` borrow.
///
/// A background task is spawned that drives the underlying `stream()`;
/// it terminates automatically when the source is fully consumed or
/// dropped.
pub fn chat_model_stream_source(
    model: Arc<dyn ChatModel>,
    messages: Vec<Message>,
    options: CallOptions,
) -> Source<Result<GenerationChunk, ProviderError>> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<GenerationChunk, ProviderError>>();
    tokio::spawn(async move {
        match model.stream(&messages, &options).await {
            Err(e) => {
                let _ = tx.send(Err(e));
            }
            Ok(mut s) => {
                while let Some(chunk) = s.next().await {
                    if tx.send(chunk).is_err() {
                        break;
                    }
                }
            }
        }
    });
    Source::from_receiver(rx)
}
