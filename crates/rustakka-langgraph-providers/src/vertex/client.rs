//! [`VertexGeminiModel`] — Vertex AI / Gemini chat client.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::EventSource;
use serde_json::Value;

use crate::error::ProviderError;
use crate::traits::ChatModel;
use crate::types::message::Message;
use crate::types::options::{CallOptions, GenerationChunk};

use super::stream::parse_gemini_sse;
use super::types::{
    messages_to_request_parts, response_to_message, GeminiFunctionDeclaration, GeminiRequest,
    GeminiResponse, GeminiTool,
};

/// Authentication source for the Vertex client.
///
/// * `Static` — caller-provided bearer token (used by tests or custom flows).
/// * `Adc` — dynamic tokens fetched via `gcp_auth` on every request.
#[derive(Clone)]
pub enum VertexAuth {
    Static(String),
    #[cfg(feature = "vertex")]
    Adc(Arc<dyn gcp_auth::TokenProvider>),
}

impl std::fmt::Debug for VertexAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VertexAuth::Static(_) => f.write_str("VertexAuth::Static(<redacted>)"),
            #[cfg(feature = "vertex")]
            VertexAuth::Adc(_) => f.write_str("VertexAuth::Adc"),
        }
    }
}

/// Vertex AI Gemini chat client.
///
/// # Examples
///
/// ```rust,ignore
/// // ADC (requires GOOGLE_APPLICATION_CREDENTIALS):
/// let auth = VertexGeminiModel::adc().await?;
/// let model = VertexGeminiModel::new("my-proj", "us-central1", "gemini-1.5-flash", auth);
///
/// // Static token (tests or custom auth flows):
/// let model = VertexGeminiModel::new(
///     "p", "us-central1", "gemini-1.5-flash",
///     VertexAuth::Static("ya29.bearer".into()),
/// );
/// ```
pub struct VertexGeminiModel {
    client: reqwest::Client,
    project: String,
    location: String,
    model: String,
    auth: VertexAuth,
    /// Defaults to `https://{location}-aiplatform.googleapis.com`.
    base_url: String,
    pub default_options: CallOptions,
}

impl std::fmt::Debug for VertexGeminiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexGeminiModel")
            .field("project", &self.project)
            .field("location", &self.location)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl VertexGeminiModel {
    pub fn new(
        project: impl Into<String>,
        location: impl Into<String>,
        model: impl Into<String>,
        auth: VertexAuth,
    ) -> Self {
        let loc = location.into();
        let base_url = format!("https://{loc}-aiplatform.googleapis.com");
        Self {
            client: reqwest::Client::new(),
            project: project.into(),
            location: loc,
            model: model.into(),
            auth,
            base_url,
            default_options: CallOptions::default(),
        }
    }

    /// Acquire an ADC-backed authentication manager (Google Application Default
    /// Credentials). Returns an error if credentials are not configured.
    #[cfg(feature = "vertex")]
    pub async fn adc() -> Result<VertexAuth, ProviderError> {
        let provider = gcp_auth::provider()
            .await
            .map_err(|e| ProviderError::Auth(e.to_string()))?;
        Ok(VertexAuth::Adc(provider))
    }

    /// Override the REST API base URL (used by tests).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
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

    async fn bearer_token(&self) -> Result<String, ProviderError> {
        match &self.auth {
            VertexAuth::Static(t) => Ok(t.clone()),
            #[cfg(feature = "vertex")]
            VertexAuth::Adc(provider) => {
                let scopes = &["https://www.googleapis.com/auth/cloud-platform"];
                let token = gcp_auth::TokenProvider::token(provider.as_ref(), scopes)
                    .await
                    .map_err(|e| ProviderError::Auth(e.to_string()))?;
                Ok(token.as_str().to_string())
            }
        }
    }

    async fn headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let bearer = self.bearer_token().await?;
        if !bearer.is_empty() {
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {bearer}")) {
                h.insert(AUTHORIZATION, v);
            }
        }
        Ok(h)
    }

    fn endpoint(&self, streaming: bool) -> String {
        let verb = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        format!(
            "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
            self.base_url, self.project, self.location, self.model, verb
        )
    }

    fn build_request(&self, messages: &[Message], options: &CallOptions) -> GeminiRequest {
        let (system_instruction, contents) = messages_to_request_parts(messages);

        let tools = if options.tools.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: options
                    .tools
                    .iter()
                    .map(GeminiFunctionDeclaration::from)
                    .collect(),
            }])
        };

        // Gemini's generationConfig uses camelCase field names.
        let mut gen_cfg = serde_json::Map::new();
        if let Some(t) = options.temperature.or(self.default_options.temperature) {
            gen_cfg.insert("temperature".into(), Value::from(t));
        }
        if let Some(m) = options.max_tokens.or(self.default_options.max_tokens) {
            gen_cfg.insert("maxOutputTokens".into(), Value::from(m));
        }
        let stops = if options.stop.is_empty() {
            self.default_options.stop.clone()
        } else {
            options.stop.clone()
        };
        if !stops.is_empty() {
            gen_cfg.insert(
                "stopSequences".into(),
                Value::Array(stops.into_iter().map(Value::String).collect()),
            );
        }
        let generation_config = if gen_cfg.is_empty() {
            None
        } else {
            Some(Value::Object(gen_cfg))
        };

        GeminiRequest {
            contents,
            system_instruction,
            tools,
            tool_config: None,
            generation_config,
        }
    }
}

#[async_trait]
impl ChatModel for VertexGeminiModel {
    async fn invoke(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Message, ProviderError> {
        let body = self.build_request(messages, options);
        let resp = self
            .client
            .post(self.endpoint(false))
            .headers(self.headers().await?)
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

        let parsed: GeminiResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;
        Ok(response_to_message(parsed))
    }

    async fn stream(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<BoxStream<'_, Result<GenerationChunk, ProviderError>>, ProviderError> {
        let body = self.build_request(messages, options);
        let request = self
            .client
            .post(self.endpoint(true))
            .headers(self.headers().await?)
            .json(&body);

        let es = EventSource::new(request).map_err(|e| ProviderError::Stream(e.to_string()))?;
        Ok(parse_gemini_sse(es))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
