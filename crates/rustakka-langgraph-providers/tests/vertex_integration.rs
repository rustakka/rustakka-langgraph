//! Integration tests for the Vertex / Gemini provider.

#![cfg(feature = "vertex")]

use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use rustakka_langgraph_providers::prelude::*;

fn model(uri: String) -> VertexGeminiModel {
    VertexGeminiModel::new(
        "test-proj",
        "us-central1",
        "gemini-1.5-flash",
        VertexAuth::Static("ya29.test".into()),
    )
    .with_base_url(uri)
}

#[tokio::test]
async fn invoke_happy_path_and_system_instruction() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-1.5-flash:generateContent",
        ))
        .and(header("authorization", "Bearer ya29.test"))
        .and(body_partial_json(json!({
            "system_instruction": {"parts": [{"text": "be helpful"}]}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hi there!"}]
                },
                "finishReason": "STOP"
            }],
            "modelVersion": "gemini-1.5-flash-001"
        })))
        .mount(&server)
        .await;

    let reply = model(server.uri())
        .invoke(
            &[Message::system("be helpful"), Message::human("hi")],
            &CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(reply.text(), "Hi there!");
}

#[tokio::test]
async fn invoke_extracts_function_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-1.5-flash:generateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}
                    ]
                },
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;

    let reply = model(server.uri())
        .invoke(
            &[Message::human("1+2?")],
            &CallOptions {
                tools: vec![ToolDefinition {
                    name: "add".into(),
                    description: "add".into(),
                    parameters: json!({"type": "object"}),
                }],
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(reply.has_tool_calls());
    assert_eq!(reply.tool_calls[0].name, "add");
    assert_eq!(reply.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
}

#[tokio::test]
async fn invoke_auth_error_maps() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
        .mount(&server)
        .await;

    let err = model(server.uri())
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, ProviderError::Auth(_)));
}

#[tokio::test]
async fn stream_sse_yields_chunks() {
    let server = MockServer::start().await;
    let sse = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n\
               data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    Mock::given(method("POST"))
        .and(path(
            "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-1.5-flash:streamGenerateContent",
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let m = model(server.uri());
    let mut s = m
        .stream(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();
    let mut collected = String::new();
    while let Some(chunk) = s.next().await {
        collected.push_str(&chunk.unwrap().text);
    }
    assert_eq!(collected, "Hello");
}
