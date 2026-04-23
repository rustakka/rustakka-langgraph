//! [`OpenAiModel`] — the primary chat-completion client.
//!
//! Supports the standard OpenAI API as well as any compatible endpoint
//! (LiteLLM, vLLM, Groq, Together, etc.) via [`OpenAiModel::with_base_url`].

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::EventSource;

use crate::error::ProviderError;
use crate::traits::ChatModel;
use crate::types::message::Message;
use crate::types::options::{CallOptions, GenerationChunk};

use super::stream::parse_sse_stream;
use super::types::{ChatCompletionRequest, ChatCompletionResponse, WireMessage, WireTool};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI chat-completion client implementing [`ChatModel`].
///
/// # Examples
///
/// ```rust,ignore
/// // Standard OpenAI
/// let model = OpenAiModel::new("sk-...", "gpt-4o");
///
/// // LiteLLM proxy
/// let model = OpenAiModel::new("anything", "gpt-4o")
///     .with_base_url("http://localhost:4000");
///
/// // vLLM / local server
/// let model = OpenAiModel::new("", "meta-llama/Llama-3-8B-Instruct")
///     .with_base_url("http://localhost:8000/v1");
/// ```
pub struct OpenAiModel {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    /// Default options merged with per-call overrides.
    pub default_options: CallOptions,
}

impl std::fmt::Debug for OpenAiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiModel")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAiModel {
    /// Create a new OpenAI client.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.into(),
            default_options: CallOptions::default(),
        }
    }

    /// Override the base URL (for LiteLLM, vLLM, Groq, etc.).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Convenience constructor for a LiteLLM proxy deployment.
    ///
    /// LiteLLM exposes the exact OpenAI chat-completions interface, so all we
    /// need to do is point at the proxy's base URL. The `api_key` may be
    /// whatever the proxy is configured to accept (often empty).
    pub fn litellm(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(api_key, model).with_base_url(base_url)
    }

    /// Convenience constructor for a vLLM / local OpenAI-compatible server.
    pub fn vllm(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("", model).with_base_url(base_url)
    }

    /// Set default options applied to every call (can be overridden per-call).
    pub fn with_default_options(mut self, opts: CallOptions) -> Self {
        self.default_options = opts;
        self
    }

    /// Provide a custom `reqwest::Client` (for custom TLS, proxies, etc.).
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !self.api_key.is_empty() {
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {}", self.api_key)) {
                h.insert(AUTHORIZATION, v);
            }
        }
        h
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn build_request(
        &self,
        messages: &[Message],
        options: &CallOptions,
        stream: bool,
    ) -> ChatCompletionRequest {
        let wire_messages: Vec<WireMessage> = messages.iter().map(WireMessage::from).collect();
        let wire_tools: Vec<WireTool> = options.tools.iter().map(WireTool::from).collect();

        ChatCompletionRequest {
            model: self.model.clone(),
            messages: wire_messages,
            temperature: options.temperature.or(self.default_options.temperature),
            max_tokens: options.max_tokens.or(self.default_options.max_tokens),
            stop: if options.stop.is_empty() {
                self.default_options.stop.clone()
            } else {
                options.stop.clone()
            },
            tools: wire_tools,
            tool_choice: options
                .tool_choice
                .clone()
                .or_else(|| self.default_options.tool_choice.clone()),
            response_format: options
                .response_format
                .clone()
                .or_else(|| self.default_options.response_format.clone()),
            stream: if stream { Some(true) } else { None },
        }
    }
}

#[async_trait]
impl ChatModel for OpenAiModel {
    async fn invoke(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Message, ProviderError> {
        let body = self.build_request(messages, options, false);
        let resp = self
            .client
            .post(self.endpoint())
            .headers(self.headers())
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(ProviderError::RateLimited { retry_after_ms: None });
            }
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ProviderError::Auth(body_text));
            }
            return Err(ProviderError::ApiError {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let response: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Parse("no choices in response".into()))?;

        Ok(choice.message.into_message())
    }

    async fn stream(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<BoxStream<'_, Result<GenerationChunk, ProviderError>>, ProviderError> {
        let body = self.build_request(messages, options, true);

        let request = self
            .client
            .post(self.endpoint())
            .headers(self.headers())
            .json(&body);

        let es = EventSource::new(request)
            .map_err(|e| ProviderError::Stream(e.to_string()))?;

        Ok(parse_sse_stream(es))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
