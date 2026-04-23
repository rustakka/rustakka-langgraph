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

- [x] **Phase 10** — Idiomatic actor + `rustakka-streams` pass. Node
      dispatch and stream fan-out now use the upstream actor & streams
      vocabulary end-to-end. Public API stayed additive; the Python
      facade was untouched.
      - [x] `rustakka-streams` pinned in the workspace manifest with a
            matching `[patch]` override for local iteration; wired into
            `rustakka-langgraph-core`, `-providers`, and `-prebuilt`.
      - [x] `NodeWorker` child actor (`coordinator/node_worker.rs`)
            with `NodeInvoke`, a `RoundRobinRouter`-backed
            `NodeDispatchRouter`, and a `OneForOneStrategy` supervisor
            that stops on panic to prevent restart storms. Replaces
            the raw `tokio::spawn` fan-out in `advance_or_finish`.
      - [x] Retries folded into the worker via
            `rustakka::pattern::BackoffOptions` (jittered exponential
            backoff); the manual `invoke_with_retry` loop was removed.
      - [x] `CoordMsg::StartRun` switched to
            `ActorRef::ask_with(.., Duration)` in `run_one_with_bus`;
            the ad-hoc `oneshot::channel` plumbing is gone.
      - [x] `StreamBus` became actor-shaped — all subscriber-list
            mutations flow through a single writer lock, while
            `publish` fans out inline to keep the existing
            `run.await` → `rx.try_recv()` ordering contract.
            `subscribe_source(modes, overflow)` returns a
            `(Subscription, Source<StreamEvent>)` pair with an
            optional `OverflowStrategy` and RAII unsubscribe.
      - [x] `runner::stream_source` (additive, alongside the legacy
            `runner::stream`) returns `(Source<StreamEvent>,
            KillSwitch, JoinHandle<GraphResult<RunResult>>)`.
            Triggering the kill switch tells the coordinator to stop
            and closes every subscriber source.
      - [x] Subgraph forwarding rewritten as
            `Source::from_receiver(child_rx).map(namespace_inject)
            .runForeach(parent_bus.publish)`.
      - [x] `ChatModel::stream_source(...)` provider helper returns a
            `Source<Result<GenerationChunk, ProviderError>>`; the ReAct
            agent's streaming path consumes it through `wire_tap`
            (publishing `OnChatModelStream` via
            `StreamWriter::chat_model_chunk`) + `Sink::collect` for
            message assembly.
      - [x] `ToolNode` fan-out replaced with
            `Source::from_iter(calls).map_async_unordered(parallelism,
            run_tool).collect(..)`; `ToolNodeOptions { parallelism }`
            is the new knob. Output order is preserved via the
            original call index.
      - [x] Bonus: `GraphValues::ensure_channel` now uses
            `DashMap::entry().or_insert_with(..)` (atomic check-and-set
            under concurrent fan-out).
      - [x] New unit tests: `NodeWorker` retry + backoff + panic
            translation, `StreamBus` subscribe / publish /
            unsubscribe / `subscribe_source` / overflow,
            `runner::stream_source` + `KillSwitch`, `ToolNode`
            parallelism + call-order preservation. Existing runner /
            coordinator / streaming / subgraph / Python suites
            continue to pass without edits.
