//! Integration tests for the OpenAI provider driven by a `wiremock` server.
//!
//! These exercise the full request-building / response-parsing path without
//! hitting the real OpenAI API, so they're safe to run in CI.

#![cfg(feature = "openai")]

use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use rustakka_langgraph_providers::prelude::*;
use rustakka_langgraph_providers::openai::OpenAiModel;

fn chat_completion_body(content: &str, tool_calls: Option<serde_json::Value>) -> serde_json::Value {
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if let Some(tcs) = tool_calls {
        message["tool_calls"] = tcs;
    }
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    })
}

#[tokio::test]
async fn invoke_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_completion_body("hello world", None)),
        )
        .mount(&server)
        .await;

    let model = OpenAiModel::new("sk-test", "gpt-4o-mini").with_base_url(server.uri());
    let reply = model
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(reply.role, rustakka_langgraph_providers::types::message::Role::Ai);
    assert_eq!(reply.text(), "hello world");
}

#[tokio::test]
async fn invoke_with_tool_calls() {
    let server = MockServer::start().await;
    let tool_calls = json!([{
        "id": "call_1",
        "type": "function",
        "function": {"name": "calc", "arguments": "{\"a\":1,\"b\":2}"}
    }]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completion_body("", Some(tool_calls))),
        )
        .mount(&server)
        .await;

    let model = OpenAiModel::new("sk-test", "gpt-4o-mini").with_base_url(server.uri());
    let opts = CallOptions {
        tools: vec![ToolDefinition {
            name: "calc".into(),
            description: "add".into(),
            parameters: json!({"type": "object"}),
        }],
        ..Default::default()
    };
    let reply = model
        .invoke(&[Message::human("2+2")], &opts)
        .await
        .unwrap();
    assert!(reply.has_tool_calls());
    assert_eq!(reply.tool_calls[0].name, "calc");
    assert_eq!(reply.tool_calls[0].arguments, json!({"a": 1, "b": 2}));
}

#[tokio::test]
async fn invoke_auth_error_maps() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(json!({"error": {"message": "bad key"}})),
        )
        .mount(&server)
        .await;

    let model = OpenAiModel::new("sk-bogus", "gpt-4o-mini").with_base_url(server.uri());
    let err = model
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, ProviderError::Auth(_)));
}

#[tokio::test]
async fn invoke_rate_limit_maps() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .mount(&server)
        .await;

    let model = OpenAiModel::new("sk-test", "gpt-4o-mini").with_base_url(server.uri());
    let err = model
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, ProviderError::RateLimited { .. }));
}

#[tokio::test]
async fn stream_assembles_sse_chunks() {
    let server = MockServer::start().await;
    // Three deltas + DONE.
    let sse_body = "data: {\"id\":\"c1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n\
                    data: {\"id\":\"c1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n\
                    data: {\"id\":\"c1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = OpenAiModel::new("sk-test", "gpt-4o-mini").with_base_url(server.uri());
    let mut stream = model
        .stream(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();

    let mut collected = String::new();
    while let Some(chunk) = stream.next().await {
        collected.push_str(&chunk.unwrap().text);
    }
    assert_eq!(collected, "Hello world");
}

#[tokio::test]
async fn stream_tool_call_deltas() {
    let server = MockServer::start().await;
    let sse_body = "data: {\"id\":\"c1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"a\\\":\"}}]},\"finish_reason\":null}]}\n\n\
                    data: {\"id\":\"c1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = OpenAiModel::new("sk-test", "gpt-4o-mini").with_base_url(server.uri());
    let mut stream = model
        .stream(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();

    let mut args_accum = String::new();
    let mut saw_name = false;
    while let Some(chunk) = stream.next().await {
        let c = chunk.unwrap();
        for tc in c.tool_call_chunks {
            if let Some(n) = tc.name {
                assert_eq!(n, "calc");
                saw_name = true;
            }
            if let Some(a) = tc.arguments {
                args_accum.push_str(&a);
            }
        }
    }
    assert!(saw_name, "first chunk should carry tool name");
    assert_eq!(args_accum, "{\"a\":1}");
}

#[tokio::test]
async fn litellm_constructor_uses_base_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_completion_body("via litellm", None)),
        )
        .mount(&server)
        .await;

    let model = OpenAiModel::litellm(server.uri(), "anything", "claude-3-haiku");
    let reply = model
        .invoke(&[Message::human("hi")], &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(reply.text(), "via litellm");
}
