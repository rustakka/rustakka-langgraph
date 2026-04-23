//! [`AzureOpenAiModel`] — reuses the OpenAI wire format with Azure's
//! deployment-scoped URL template and `api-key` authentication header.

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::EventSource;

use crate::error::ProviderError;
use crate::openai::{parse_sse_stream, ChatCompletionRequest, ChatCompletionResponse, WireMessage, WireTool};
use crate::traits::ChatModel;
use crate::types::message::Message;
use crate::types::options::{CallOptions, GenerationChunk};

const DEFAULT_API_VERSION: &str = "2024-02-15-preview";

/// Azure OpenAI deployment client.
///
/// # Examples
///
/// ```rust,ignore
/// let model = AzureOpenAiModel::new(
///     "https://my-aoai.openai.azure.com",
///     "my-gpt4o-deployment",
///     "xxxxxxxxxxxx",
/// );
/// ```
pub struct AzureOpenAiModel {
    client: reqwest::Client,
    endpoint: String,
    deployment_id: String,
    api_version: String,
    api_key: String,
    /// Deployment ID used as the "model name" for logging.
    model: String,
    pub default_options: CallOptions,
}

impl std::fmt::Debug for AzureOpenAiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAiModel")
            .field("endpoint", &self.endpoint)
            .field("deployment_id", &self.deployment_id)
            .field("api_version", &self.api_version)
            .finish()
    }
}

impl AzureOpenAiModel {
    pub fn new(
        endpoint: impl Into<String>,
        deployment_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let deployment = deployment_id.into();
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            deployment_id: deployment.clone(),
            api_version: DEFAULT_API_VERSION.into(),
            api_key: api_key.into(),
            model: deployment,
            default_options: CallOptions::default(),
        }
    }

    pub fn with_api_version(mut self, v: impl Into<String>) -> Self {
        self.api_version = v.into();
        self
    }

    pub fn with_default_options(mut self, opts: CallOptions) -> Self {
        self.default_options = opts;
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Ok(v) = HeaderValue::from_str(&self.api_key) {
            h.insert("api-key", v);
        }
        h
    }

    fn endpoint_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint, self.deployment_id, self.api_version
        )
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
            model: self.deployment_id.clone(),
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
impl ChatModel for AzureOpenAiModel {
    async fn invoke(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Message, ProviderError> {
        let body = self.build_request(messages, options, false);
        let resp = self
            .client
            .post(self.endpoint_url())
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
            .post(self.endpoint_url())
            .headers(self.headers())
            .json(&body);

        let es = EventSource::new(request).map_err(|e| ProviderError::Stream(e.to_string()))?;
        Ok(parse_sse_stream(es))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
