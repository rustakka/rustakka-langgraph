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
