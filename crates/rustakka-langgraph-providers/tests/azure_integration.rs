//! Integration tests for the Azure OpenAI provider.

#![cfg(feature = "azure")]

use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use rustakka_langgraph_providers::prelude::*;

fn body(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-az",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-dep",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
}

#[tokio::test]
async fn invoke_uses_api_key_header_and_deployment_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/my-dep/chat/completions"))
        .and(query_param("api-version", "2024-02-15-preview"))
        .and(header("api-key", "top-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body("hi from azure")))
        .mount(&server)
        .await;

    let model = AzureOpenAiModel::new(server.uri(), "my-dep", "top-secret");
    let reply = model
        .invoke(&[Message::human("hello")], &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(reply.text(), "hi from azure");
}

#[tokio::test]
async fn invoke_respects_custom_api_version() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/my-dep/chat/completions"))
        .and(query_param("api-version", "2024-09-01-preview"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body("v2")))
        .mount(&server)
        .await;

    let model = AzureOpenAiModel::new(server.uri(), "my-dep", "k")
        .with_api_version("2024-09-01-preview");
    let reply = model
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(reply.text(), "v2");
}

#[tokio::test]
async fn stream_reuses_openai_sse_parser() {
    let server = MockServer::start().await;
    let sse = "data: {\"id\":\"c1\",\"model\":\"gpt-4o-dep\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi \"},\"finish_reason\":null}]}\n\n\
               data: {\"id\":\"c1\",\"model\":\"gpt-4o-dep\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"azure\"},\"finish_reason\":\"stop\"}]}\n\n\
               data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/openai/deployments/my-dep/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = AzureOpenAiModel::new(server.uri(), "my-dep", "k");
    let mut s = model
        .stream(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();
    let mut out = String::new();
    while let Some(chunk) = s.next().await {
        out.push_str(&chunk.unwrap().text);
    }
    assert_eq!(out, "Hi azure");
}
