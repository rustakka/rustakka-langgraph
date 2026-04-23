//! [`BedrockModel`] — AWS Bedrock chat client using the Converse API.

use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_bedrockruntime::types as br;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::traits::ChatModel;
use crate::types::message::Message;
use crate::types::options::{CallOptions, GenerationChunk, ToolCallChunkDelta};

use super::adapters::BedrockAdapter;
use super::convert::{
    converse_output_to_message, document_to_json, messages_to_converse, tools_to_converse,
};

/// AWS Bedrock chat client backed by the Converse API.
///
/// ```rust,ignore
/// let config = aws_config::load_from_env().await;
/// let model = BedrockModel::new(&config, "anthropic.claude-3-haiku-20240307-v1:0");
/// let reply = model.invoke(&msgs, &CallOptions::default()).await?;
/// ```
pub struct BedrockModel {
    client: BedrockClient,
    model_id: String,
    /// Optional legacy adapter for `InvokeModel` (bypasses Converse).
    adapter: Option<Arc<dyn BedrockAdapter>>,
    pub default_options: CallOptions,
}

impl std::fmt::Debug for BedrockModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockModel")
            .field("model_id", &self.model_id)
            .field("adapter", &self.adapter.is_some())
            .finish()
    }
}

impl BedrockModel {
    /// Build a Bedrock client from a shared AWS config.
    pub fn new(config: &aws_config::SdkConfig, model_id: impl Into<String>) -> Self {
        Self {
            client: BedrockClient::new(config),
            model_id: model_id.into(),
            adapter: None,
            default_options: CallOptions::default(),
        }
    }

    /// Use a pre-built SDK client directly (useful for test overrides).
    pub fn with_client(mut self, client: BedrockClient) -> Self {
        self.client = client;
        self
    }

    /// Swap in a legacy-payload adapter. The client then uses `InvokeModel`
    /// instead of Converse.
    pub fn with_adapter(mut self, adapter: Arc<dyn BedrockAdapter>) -> Self {
        self.adapter = Some(adapter);
        self
    }

    pub fn with_default_options(mut self, opts: CallOptions) -> Self {
        self.default_options = opts;
        self
    }

    fn merged_inference_config(
        &self,
        options: &CallOptions,
    ) -> Option<br::InferenceConfiguration> {
        let temp = options.temperature.or(self.default_options.temperature);
        let max = options.max_tokens.or(self.default_options.max_tokens);
        let stops = if options.stop.is_empty() {
            self.default_options.stop.clone()
        } else {
            options.stop.clone()
        };
        if temp.is_none() && max.is_none() && stops.is_empty() {
            return None;
        }
        let mut b = br::InferenceConfiguration::builder();
        if let Some(t) = temp {
            b = b.temperature(t);
        }
        if let Some(m) = max {
            b = b.max_tokens(m as i32);
        }
        if !stops.is_empty() {
            b = b.set_stop_sequences(Some(stops));
        }
        Some(b.build())
    }

    async fn invoke_via_adapter(
        &self,
        adapter: &dyn BedrockAdapter,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Message, ProviderError> {
        let body = adapter.build_body(messages, options)?;
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| ProviderError::Parse(format!("serialize body: {e}")))?;

        let resp = self
            .client
            .invoke_model()
            .model_id(self.model_id.clone())
            .body(aws_smithy_types::Blob::new(bytes))
            .content_type("application/json")
            .accept("application/json")
            .send()
            .await
            .map_err(|e| ProviderError::Other(format!("bedrock invoke_model: {e}")))?;

        let body_bytes = resp.body.into_inner();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| ProviderError::Parse(e.to_string()))?;
        adapter.parse_response(&v)
    }
}

#[async_trait]
impl ChatModel for BedrockModel {
    async fn invoke(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<Message, ProviderError> {
        if let Some(adapter) = &self.adapter {
            return self
                .invoke_via_adapter(adapter.as_ref(), messages, options)
                .await;
        }

        let (system, turns) = messages_to_converse(messages)?;
        let tool_config = tools_to_converse(&options.tools)?;

        let mut req = self.client.converse().model_id(self.model_id.clone());
        req = req.set_messages(Some(turns));
        if !system.is_empty() {
            req = req.set_system(Some(system));
        }
        if let Some(cfg) = self.merged_inference_config(options) {
            req = req.inference_config(cfg);
        }
        if let Some(tc) = tool_config {
            req = req.tool_config(tc);
        }

        let out = req
            .send()
            .await
            .map_err(|e| ProviderError::Other(format!("bedrock converse: {e}")))?;

        let output = out
            .output
            .ok_or_else(|| ProviderError::Parse("bedrock converse: missing output".into()))?;
        let br::ConverseOutput::Message(msg) = output else {
            return Err(ProviderError::Parse(
                "bedrock converse: unexpected output variant".into(),
            ));
        };
        Ok(converse_output_to_message(&msg))
    }

    async fn stream(
        &self,
        messages: &[Message],
        options: &CallOptions,
    ) -> Result<BoxStream<'_, Result<GenerationChunk, ProviderError>>, ProviderError> {
        if self.adapter.is_some() {
            return Err(ProviderError::other(
                "BedrockAdapter-based streaming is not supported",
            ));
        }

        let (system, turns) = messages_to_converse(messages)?;
        let tool_config = tools_to_converse(&options.tools)?;

        let mut req = self
            .client
            .converse_stream()
            .model_id(self.model_id.clone());
        req = req.set_messages(Some(turns));
        if !system.is_empty() {
            req = req.set_system(Some(system));
        }
        if let Some(cfg) = self.merged_inference_config(options) {
            req = req.inference_config(cfg);
        }
        if let Some(tc) = tool_config {
            req = req.tool_config(tc);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::Other(format!("bedrock converse_stream: {e}")))?;

        let mut event_stream = resp.stream;

        let s = async_stream::stream! {
            let mut active_tool: Option<(usize, String)> = None; // (index, name)

            loop {
                match event_stream.recv().await {
                    Ok(Some(event)) => {
                        match event {
                            br::ConverseStreamOutput::ContentBlockStart(ev) => {
                                if let Some(start) = ev.start {
                                    if let br::ContentBlockStart::ToolUse(tu) = start {
                                        let idx = ev.content_block_index as usize;
                                        active_tool = Some((idx, tu.name.clone()));
                                        yield Ok(GenerationChunk {
                                            tool_call_chunks: vec![ToolCallChunkDelta {
                                                index: idx,
                                                id: Some(tu.tool_use_id.clone()),
                                                name: Some(tu.name),
                                                arguments: None,
                                            }],
                                            ..Default::default()
                                        });
                                    }
                                }
                            }
                            br::ConverseStreamOutput::ContentBlockDelta(ev) => {
                                if let Some(delta) = ev.delta {
                                    match delta {
                                        br::ContentBlockDelta::Text(t) => {
                                            yield Ok(GenerationChunk::text(t));
                                        }
                                        br::ContentBlockDelta::ToolUse(tu) => {
                                            let idx = ev.content_block_index as usize;
                                            let name = active_tool
                                                .as_ref()
                                                .filter(|(i, _)| *i == idx)
                                                .map(|(_, n)| n.clone());
                                            yield Ok(GenerationChunk {
                                                tool_call_chunks: vec![ToolCallChunkDelta {
                                                    index: idx,
                                                    id: None,
                                                    name,
                                                    arguments: Some(tu.input),
                                                }],
                                                ..Default::default()
                                            });
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            br::ConverseStreamOutput::MessageStop(ev) => {
                                yield Ok(GenerationChunk {
                                    metadata: serde_json::json!({
                                        "stop_reason": format!("{:?}", ev.stop_reason),
                                    }),
                                    ..Default::default()
                                });
                            }
                            br::ConverseStreamOutput::Metadata(_)
                            | br::ConverseStreamOutput::MessageStart(_)
                            | br::ConverseStreamOutput::ContentBlockStop(_) => {}
                            _ => {}
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        yield Err(ProviderError::Stream(e.to_string()));
                        break;
                    }
                }
            }
            // reference to silence unused-var lint when no tool-use was seen
            let _ = &active_tool;
            let _ = document_to_json; // keep import warning-free across feature combos
        };

        Ok(Box::pin(s))
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }
}
