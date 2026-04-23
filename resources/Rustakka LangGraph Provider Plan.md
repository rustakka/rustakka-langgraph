# **Provider Implementation Plan for rustakka-langgraph**

This document outlines the architectural design and phased implementation plan for adding comprehensive LLM provider support (OpenAI, LiteLLM, Ollama, Cloud Providers, etc.) to rustakka-langgraph.

## **1\. Core Architecture & Abstractions**

Before implementing specific APIs, we must define the standard interfaces. Inspired by LangChain/LangGraph but tailored for Rust's strong typing and rustakka's asynchronous actor model.

### **1.1 Message & Content Types**

We need a unified representation of dialogue and multimodality that all providers will consume and produce.

* **Message Enum:** SystemMessage, HumanMessage, AiMessage, ToolMessage.  
* **Content Enum:** Text(String), ImageUrl(String), ToolCallRequest(...).  
* **Structured Outputs:** Utilizing serde\_json::Value for tool schemas and arguments.

### **1.2 The ChatModel Trait**

The core interface every provider must implement. It must be Send \+ Sync to move across actor boundaries safely.  
\#\[async\_trait\]  
pub trait ChatModel: Send \+ Sync \+ std::fmt::Debug {  
    /// Standard blocking/async invocation  
    async fn invoke(  
        \&self,   
        messages: &\[Message\],   
        options: \&CallOptions  
    ) \-\> Result\<AiMessage, ProviderError\>;

    /// Streaming response returning an async Stream of chunks  
    async fn stream(  
        \&self,   
        messages: &\[Message\],   
        options: \&CallOptions  
    ) \-\> Result\<BoxStream\<'\_, Result\<GenerationChunk, ProviderError\>\>, ProviderError\>;  
}

### **1.3 rustakka Actor Integration**

LLM network calls are I/O bound and can have high latency. To prevent blocking the rustakka actor dispatchers:

1. **Direct Future Spawning:** Graph Node actors can use tokio::spawn to call the LLM and then pipe the result back to the actor via a message (e.g., ctx.myself().tell(LlmResponse(result))).  
2. **Provider Actors:** Alternatively, wrap ChatModel inside a dedicated LlmActor that processes InvokePrompt messages and responds with PromptResult messages, allowing rate-limiting and connection pooling at the actor level.

## **2\. Implementation Phases**

### **Phase 1: The Standard OpenAI Interface (The Keystone)**

Because many tools (LiteLLM, vLLM, local inference servers) mimic the OpenAI API, building a robust OpenAI client provides the highest initial ROI.

* **Crate Dependency:** reqwest for HTTP, reqwest-eventsource for Server-Sent Events (SSE) streaming, serde for JSON.  
* **Features:**  
  * /v1/chat/completions endpoint mapping.  
  * **Tool Calling:** Mapping Rust structs to JSON Schema for the tools array.  
  * **Streaming:** Yielding GenerationChunk by parsing SSE data: {...} lines.  
* **LiteLLM Support:** \* LiteLLM uses the exact same interface as OpenAI. Support is achieved simply by allowing configuration of a custom base\_url and dropping auth validation requirements.  
  * *Implementation:* OpenAiModel::with\_base\_url("http://localhost:4000")

### **Phase 2: Local & Open Source Models (Ollama)**

Ollama is the standard for local development and offline agent testing.

* **API Differences:** Ollama has its own /api/chat and /api/generate endpoints.  
* **Features:**  
  * Map rustakka messages to Ollama's message JSON structure.  
  * Implement NDJSON (Newline Delimited JSON) streaming parser, as Ollama streams individual JSON objects rather than using standard SSE.  
  * Support for the new native Tool Calling feature introduced in recent Ollama versions.

### **Phase 3: Cloud Providers (Enterprise Focus)**

Enterprise users of LangGraph often require specific cloud providers for data privacy and IAM integration.

* **AWS Bedrock (aws\_bedrock\_model)**  
  * *Dependency:* aws-sdk-bedrockruntime.  
  * *Challenge:* Bedrock has different payload shapes for Anthropic (Claude 3 uses Messages API), Meta (Llama 3), and Amazon Titan.  
  * *Solution:* Implement a BedrockAdapter trait that transforms standard Message objects into the specific JSON shape required by the chosen model family within Bedrock.  
* **Azure OpenAI (azure\_openai\_model)**  
  * Can largely reuse the core logic from Phase 1\.  
  * *Differences:* Requires api-key in a different header (api-key instead of Authorization: Bearer), and URLs include the deployment name and API version (/openai/deployments/{deployment\_id}/chat/completions?api-version=2023-12-01-preview).  
* **GCP Vertex AI (gcp\_vertex\_model)**  
  * Focus on the Gemini API (generateContent and streamGenerateContent).  
  * Requires mapping standard tool calls to Gemini's functionDeclarations.  
  * Use GCP Application Default Credentials (ADC) for authentication via gcp\_auth or similar Rust crates.

### **Phase 4: Streaming Engine Integration with LangGraph**

Streaming in LangGraph is complex because you are streaming *node updates* and *LLM tokens* simultaneously.

* **Design:** Implement a StreamHandler callback or channel in CallOptions.  
* When a Node Actor invokes the LLM, it passes a channel sender (mpsc::Sender\<Token\>).  
* As the LLM streams, the Node Actor pushes these tokens to a main event bus (e.g., rustakka PubSub), allowing frontends to render UI updates in real-time while the graph is still processing.

## **3\. Directory Structure**

A recommended module layout for rustakka-langgraph:  
rustakka-langgraph/  
├── src/  
│   ├── providers/  
│   │   ├── mod.rs               \# Exports all providers  
│   │   ├── traits.rs            \# ChatModel, EmbeddingModel traits  
│   │   ├── messages.rs          \# SystemMessage, HumanMessage, ToolCall  
│   │   ├── openai/  
│   │   │   ├── client.rs        \# API logic, Tool schema mapping  
│   │   │   └── stream.rs        \# SSE parsing  
│   │   ├── ollama/  
│   │   │   └── client.rs        \# NDJSON streaming, local routing  
│   │   ├── bedrock/  
│   │   │   ├── client.rs        \# AWS SDK wrapper  
│   │   │   └── adapters.rs      \# Claude 3 vs Llama 3 payload shapers  
│   │   └── vertex/  
│   │       └── client.rs        \# Gemini integration  
│   ├── graph/  
│   │   └── node\_actor.rs        \# How rustakka actors consume the providers

## **4\. Progress Checklist**

Status of the action items from the original plan, updated as delivery lands in `crates/rustakka-langgraph-providers` and `crates/rustakka-langgraph-prebuilt`.

- [x] **Define Core Traits** — `ChatModel`, `Message`, `ContentBlock`, `ToolCallRequest`, `CallOptions`, `GenerationChunk`, `ToolCallChunkDelta`, and `ProviderError` live under `crates/rustakka-langgraph-providers/src/{traits.rs,types/,error.rs}`.
- [x] **Mock Provider** — `MockChatModel` (feature `mock`) with deterministic invoke/stream FIFO behaviour.
- [x] **OpenAI + LiteLLM** — `OpenAiModel` with full chat-completion, SSE streaming, tool-calling; `OpenAiModel::litellm(...)` and `::vllm(...)` convenience constructors. Covered by `tests/openai_integration.rs` (wiremock: happy path, tool round-trip, SSE text + tool-call deltas, 401/429 mapping, LiteLLM base URL).
- [x] **Ollama** — `OllamaModel` at `src/ollama/` with `/api/chat` (NDJSON streaming, native tool-calling). Covered by `tests/ollama_integration.rs`.
- [x] **Vertex AI / Gemini** — `VertexGeminiModel` at `src/vertex/` using `generateContent` + `streamGenerateContent?alt=sse`; ADC via `gcp_auth::provider()` or `VertexAuth::Static` for tests. Covered by `tests/vertex_integration.rs`.
- [x] **AWS Bedrock** — `BedrockModel` at `src/bedrock/` using the Converse / ConverseStream API (normalizes Claude / Llama / Mistral / Titan); optional `BedrockAdapter` (with a `Claude3Adapter` reference impl) for legacy `InvokeModel` flows. Pure conversion unit tests in `convert.rs` and `adapters.rs`.
- [x] **Azure OpenAI** — `AzureOpenAiModel` at `src/azure/client.rs` reuses the OpenAI wire format and SSE parser; only swaps in the `api-key` header and deployment URL template. Covered by `tests/azure_integration.rs`.
- [x] **Tool Calling Engine** — existing `ToolNode` / `tools_condition` in `rustakka-langgraph-prebuilt` unchanged; `chat_model_fn` (in `providers_adapter.rs`, feature `providers`) bridges any `ChatModel` to the ReAct `ModelFn`, mapping the `{"role","content","tool_calls":[{"id","name","args"}]}` shape in graph state to/from `Message`. End-to-end coverage in `tests/react_agent_with_mock_provider.rs` for both `InvocationMode::Invoke` and `InvocationMode::Stream`.
- [x] **Streaming → StreamBus Integration** — `chat_model_fn` in `Stream` mode publishes each `GenerationChunk` to the currently installed `StreamWriter` as a `StreamEvent::Messages` before assembling the final `Message`. Assembly accumulates text + tool-call argument deltas across chunks.

### Out of scope (tracked for follow-up)

- `EmbeddingModel` trait + providers.
- Dedicated `LlmActor` for rate-limiting / connection-pooling — deferred until a concrete backpressure requirement exists; current `ChatModel` is already `Send + Sync` and actor-safe.
- Python bindings for the providers crate in `crates/py-bindings/` — deferred until the Rust surface is stable.