//! Integration tests for the Ollama provider.

#![cfg(feature = "ollama")]

use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use rustakka_langgraph_providers::ollama::OllamaModel;
use rustakka_langgraph_providers::prelude::*;

#[tokio::test]
async fn invoke_non_streaming_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(json!({"stream": false, "model": "llama3:8b"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "llama3:8b",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "hi from ollama"},
            "done": true
        })))
        .mount(&server)
        .await;

    let model = OllamaModel::new("llama3:8b").with_base_url(server.uri());
    let reply = model
        .invoke(&[Message::human("hello")], &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(reply.text(), "hi from ollama");
}

#[tokio::test]
async fn invoke_with_tool_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "llama3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}
                ]
            },
            "done": true
        })))
        .mount(&server)
        .await;

    let model = OllamaModel::new("llama3:8b").with_base_url(server.uri());
    let opts = CallOptions {
        tools: vec![ToolDefinition {
            name: "add".into(),
            description: "adds".into(),
            parameters: json!({"type": "object"}),
        }],
        ..Default::default()
    };
    let reply = model
        .invoke(&[Message::human("add 1 and 2")], &opts)
        .await
        .unwrap();
    assert!(reply.has_tool_calls());
    assert_eq!(reply.tool_calls[0].name, "add");
    assert_eq!(reply.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
}

#[tokio::test]
async fn stream_parses_ndjson_chunks() {
    let server = MockServer::start().await;
    // Three NDJSON frames, last one `done: true`.
    let ndjson = concat!(
        r#"{"model":"llama3:8b","message":{"role":"assistant","content":"Hel"},"done":false}"#,
        "\n",
        r#"{"model":"llama3:8b","message":{"role":"assistant","content":"lo"},"done":false}"#,
        "\n",
        r#"{"model":"llama3:8b","message":{"role":"assistant","content":" there"},"done":true}"#,
        "\n",
    );
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(ndjson.as_bytes().to_vec(), "application/x-ndjson"),
        )
        .mount(&server)
        .await;

    let model = OllamaModel::new("llama3:8b").with_base_url(server.uri());
    let mut stream = model
        .stream(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();

    let mut collected = String::new();
    while let Some(chunk) = stream.next().await {
        collected.push_str(&chunk.unwrap().text);
    }
    assert_eq!(collected, "Hello there");
}

#[tokio::test]
async fn stream_tool_call_delta() {
    let server = MockServer::start().await;
    let ndjson = concat!(
        r#"{"model":"llama3:8b","message":{"role":"assistant","tool_calls":[{"function":{"name":"calc","arguments":{"x":7}}}]},"done":false}"#,
        "\n",
        r#"{"model":"llama3:8b","message":{"role":"assistant","content":""},"done":true}"#,
        "\n",
    );
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(ndjson.as_bytes().to_vec(), "application/x-ndjson"),
        )
        .mount(&server)
        .await;

    let model = OllamaModel::new("llama3:8b").with_base_url(server.uri());
    let mut stream = model
        .stream(&[Message::human("calc 7")], &CallOptions::default())
        .await
        .unwrap();

    let mut saw = false;
    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        for tc in c.tool_call_chunks {
            assert_eq!(tc.name.as_deref(), Some("calc"));
            assert!(tc.arguments.as_deref().unwrap().contains("\"x\":7"));
            saw = true;
        }
    }
    assert!(saw, "should have observed at least one tool-call delta");
}

#[tokio::test]
async fn invoke_propagates_http_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    let model = OllamaModel::new("llama3:8b").with_base_url(server.uri());
    let err = model
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap_err();
    match err {
        ProviderError::ApiError { status, .. } => assert_eq!(status, 500),
        other => panic!("unexpected: {other:?}"),
    }
}
