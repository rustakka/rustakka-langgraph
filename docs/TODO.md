# Implementation progress

Phases mirror the plan at `.cursor/plans/rust-langgraph_on_rustakka_*.plan.md`.
Update the checkboxes after each PR.

- [x] **Phase 0** — Workspace + rustakka git dep + maturin + dev scripts + CI
- [x] **Phase 1** — Native Pregel engine (channels, state, GraphCoordinator,
      NodeActor, invoke/stream)
- [x] **Phase 2** — Control-flow parity (Command, Send fan-out, interrupt/resume,
      subgraphs)
- [x] **Phase 3** — Checkpointers: trait + MemorySaver + SqliteSaver +
      Postgres/AsyncPostgresSaver with upstream schema parity
- [x] **Phase 4** — BaseStore + InMemoryStore + PostgresStore (sync+async, TTL)
- [x] **Phase 5** — StreamBus actor + values/updates/messages/custom/debug modes
      + `StreamWriter` task-local for Custom/Messages emission from nodes
- [x] **Phase 6** — PyO3 bindings (`pylanggraph` cdylib) + `python/langgraph`
      facade + `rustakka_langgraph` package with upstream-compatible imports
- [x] **Phase 7** — Prebuilt helpers (`create_react_agent`, `ToolNode`,
      `tools_condition`) + functional API (`@entrypoint`, `@task`)
- [x] **Phase 8** — Rust per-crate tests (30 passing), Python parity tests
      (11 passing via pytest + maturin), profiler scenarios:
      `invoke` / `stream` / `fanout` / `checkpoint-heavy`
- [x] **Phase 9** — Feature-gap closure against upstream LangGraph v0.4+:
      - [x] Declared-but-ignored knobs wired through the coordinator:
            `RetryPolicy`, `interrupt_before` / `interrupt_after`,
            compile-time `recursion_limit`, `Command { graph: PARENT }`,
            `Checkpointer::put_writes`.
      - [x] State-inspection API: `runner::get_state` /
            `get_state_history` / `update_state` + Python
            `CompiledStateGraph.get_state` / `.get_state_history` /
            `.update_state` / `.with_config`.
      - [x] Streaming parity: namespaced events, `stream_mode=list`,
            `subgraphs=True`, and `astream_events` v2 event kinds
            (`on_chain_*`, `on_chat_model_stream`, `on_tool_*`).
      - [x] `BaseStore` injection via task-local `CURRENT_STORE`, with
            `invoke_with_store` in the Rust runner and
            `PyCompiledStateGraph.attach_store` plumbed through Python.
            `InMemoryStore::with_embedder(...)` adds cosine-similarity
            semantic search.
      - [x] `CachePolicy` (per-node input caching with TTL) and
            `Durability { Sync | Async | Exit }` checkpoint flush mode.
      - [x] Prebuilt multi-agent factories: `create_supervisor` and
            `create_swarm`.
      - [x] Graph visualization: `CompiledStateGraph::draw_mermaid()` /
            `draw_ascii()` (Rust + Python).

The nine-phase implementation plan is now complete. Remaining gaps
(structured-output `response_format` on ReAct, JSON-schema tool derive
macros, `get_stream_writer()` / `get_store()` task-locals surfaced to
Python, pluggable `SerializerProtocol`) are tracked as follow-ups in
the plan under `.cursor/plans/`.
