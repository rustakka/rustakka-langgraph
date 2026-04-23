//! [`OllamaModel`] — chat client for a local (or remote) Ollama server.
//!
//! Talks to the `/api/chat` endpoint and uses NDJSON for streaming. Tool
//! calling mirrors OpenAI's JSON-Schema shape, which Ollama 0.3+ accepts
//! natively.

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProviderError;
use crate::traits::ChatModel;
use crate::types::message::{ContentBlock, Message, Role, ToolCallRequest};
use crate::types::options::{CallOptions, GenerationChunk, ToolDefinition};

use super::stream::parse_ndjson_stream;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Chat client for a local Ollama server.
///
/// # Examples
///
/// ```rust,ignore
/// let model = OllamaModel::new("llama3:8b");
/// let reply = model.invoke(&msgs, &CallOptions::default()).await?;
/// ```
pub struct OllamaModel {
    client: reqwest::Client,
    model: String,
    base_url: String,
    pub default_options: CallOptions,
}

impl std::fmt::Debug for OllamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaModel")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OllamaModel {
    /// Create a new Ollama client pointing at `http://localhost:11434`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.into(),
            default_options: CallOptions::default(),
        }
    }

    /// Override the base URL (e.g. a remote Ollama instance).
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

    fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h
    }

    fn endpoint(&self) -> String {
        format!("{}/api/chat", self.base_url)
    }

    fn build_request(
        &self,
        messages: &[Message],
        options: &CallOptions,
        stream: bool,
    ) -> OllamaChatRequest {
        let wire_messages: Vec<OllamaMessage> =
            messages.iter().map(OllamaMessage::from).collect();
        let wire_tools: Vec<OllamaTool> = options.tools.iter().map(OllamaTool::from).collect();

        // Merge CallOptions.temperature / max_tokens / stop into Ollama's
        // `options` bag (Ollama uses `num_predict` rather than `max_tokens`).
        let mut opts_bag = serde_json::Map::new();
        if let Some(t) = options.temperature.or(self.default_options.temperature) {
            opts_bag.insert("temperature".into(), Value::from(t));
        }
        if let Some(m) = options.max_tokens.or(self.default_options.max_tokens) {
            opts_bag.insert("num_predict".into(), Value::from(m));
        }
        let stops = if options.stop.is_empty() {
            self.default_options.stop.clone()
        } else {
            options.stop.clone()
        };
        if !stops.is_empty() {
            opts_bag.insert("stop".into(), Value::Array(stops.into_iter().map(Value::String).collect()));
        }

        OllamaChatRequest {
            model: self.model.clone(),
            messages: wire_messages,
            stream,
            tools: wire_tools,
            options: if opts_bag.is_empty() {
                None
            } else {
                Some(Value::Object(opts_bag))
            },
        }
    }
}

#[async_trait]
impl ChatModel for OllamaModel {
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
            return Err(ProviderError::ApiError {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let response: OllamaChatResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(response.into_message())
    }

    async fn stream(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<BoxStream<'_, Result<GenerationChunk, ProviderError>>, ProviderError> {
        let body = self.build_request(messages, options, true);
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
            return Err(ProviderError::ApiError {
                status: status.as_u16(),
                body: body_text,
            });
        }

        Ok(parse_ndjson_stream(resp))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(crate) struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OllamaTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Base64-encoded image bytes (Ollama's multimodal input shape).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<OllamaFunctionCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OllamaToolFunction,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaChatResponse {
    pub model: String,
    #[serde(default)]
    pub message: Option<OllamaMessage>,
    #[serde(default)]
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<&Message> for OllamaMessage {
    fn from(msg: &Message) -> Self {
        // Collect text and image-url blocks separately. Ollama expects raw
        // base64 bytes in `images`; we best-effort forward the URL string and
        // let callers supply already-base64 content if they need it.
        let mut text = String::new();
        let mut images: Vec<String> = Vec::new();
        for block in &msg.content {
            match block {
                ContentBlock::Text { text: t } => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
                ContentBlock::ImageUrl { url, .. } => {
                    images.push(url.clone());
                }
            }
        }

        let tool_calls = if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| OllamaToolCall {
                        function: Some(OllamaFunctionCall {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        }),
                    })
                    .collect(),
            )
        };

        OllamaMessage {
            role: msg.role.as_openai_str().to_string(),
            content: if text.is_empty() { None } else { Some(text) },
            images,
            tool_calls,
        }
    }
}

impl OllamaChatResponse {
    fn into_message(self) -> Message {
        let wire = self.message.unwrap_or(OllamaMessage {
            role: "assistant".into(),
            content: None,
            images: Vec::new(),
            tool_calls: None,
        });

        let role = Role::from_openai_str(&wire.role).unwrap_or(Role::Ai);
        let content = match wire.content {
            Some(s) if !s.is_empty() => vec![ContentBlock::Text { text: s }],
            _ => Vec::new(),
        };

        let tool_calls: Vec<ToolCallRequest> = wire
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .filter_map(|tc| {
                let f = tc.function?;
                Some(ToolCallRequest {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: f.name,
                    arguments: f.arguments,
                })
            })
            .collect();

        Message {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }
}

impl From<&ToolDefinition> for OllamaTool {
    fn from(td: &ToolDefinition) -> Self {
        OllamaTool {
            tool_type: "function".into(),
            function: OllamaToolFunction {
                name: td.name.clone(),
                description: td.description.clone(),
                parameters: td.parameters.clone(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wire_message_from_text_message() {
        let m = Message::human("hello");
        let w: OllamaMessage = (&m).into();
        assert_eq!(w.role, "user");
        assert_eq!(w.content.as_deref(), Some("hello"));
        assert!(w.images.is_empty());
    }

    #[test]
    fn wire_message_collects_multimodal_blocks() {
        let m = Message::human_multimodal(vec![
            ContentBlock::Text { text: "What is this?".into() },
            ContentBlock::ImageUrl {
                url: "data:image/png;base64,abc".into(),
                detail: None,
            },
        ]);
        let w: OllamaMessage = (&m).into();
        assert_eq!(w.content.as_deref(), Some("What is this?"));
        assert_eq!(w.images, vec!["data:image/png;base64,abc".to_string()]);
    }

    #[test]
    fn into_message_extracts_tool_calls() {
        let resp = OllamaChatResponse {
            model: "llama3:8b".into(),
            message: Some(OllamaMessage {
                role: "assistant".into(),
                content: Some("calling calc".into()),
                images: Vec::new(),
                tool_calls: Some(vec![OllamaToolCall {
                    function: Some(OllamaFunctionCall {
                        name: "calc".into(),
                        arguments: json!({"a": 1, "b": 2}),
                    }),
                }]),
            }),
            done: true,
        };
        let msg = resp.into_message();
        assert_eq!(msg.role, Role::Ai);
        assert_eq!(msg.text(), "calling calc");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "calc");
        assert_eq!(msg.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn request_serializes_stream_and_options() {
        let model = OllamaModel::new("llama3:8b");
        let opts = CallOptions {
            temperature: Some(0.2),
            max_tokens: Some(64),
            stop: vec!["STOP".into()],
            ..Default::default()
        };
        let req = model.build_request(&[Message::human("hi")], &opts, true);
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["stream"], json!(true));
        assert_eq!(v["model"], json!("llama3:8b"));
        let temp = v["options"]["temperature"].as_f64().unwrap();
        assert!((temp - 0.2).abs() < 1e-6, "temperature: {temp}");
        assert_eq!(v["options"]["num_predict"], json!(64));
        assert_eq!(v["options"]["stop"], json!(["STOP"]));
    }
}
