# Architecture

`rustakka-langgraph` is a faithful Pregel/BSP runtime for
[LangGraph](https://github.com/langchain-ai/langgraph) written in Rust on
top of the [rustakka](https://github.com/cognect/rustakka) actor system.
Python remains the *authoring* surface (via PyO3 bindings) while the
*engine* — planning, dispatch, channels, checkpointing, streaming —
executes GIL-free in Rust.

This document explains how those pieces fit together so that contributors
can reason about correctness and extend the system without breaking
parity with upstream LangGraph.

> **Parity status.** Phase 9 (see [TODO.md](TODO.md)) closes the last
> large upstream gaps identified in the Phase 8 review: static breakpoints
> (`interrupt_before` / `interrupt_after`), pending-writes durability,
> `Command { graph: PARENT }` subgraph escalation, state inspection
> (`get_state` / `get_state_history` / `update_state` / `with_config`),
> multi-mode + `subgraphs=True` streaming, the `astream_events` v2 event
> kinds, `BaseStore` injection into nodes, semantic search via `Embedder`,
> `CachePolicy`, `Durability`, `create_supervisor` / `create_swarm`, and
> Mermaid / ASCII visualization helpers.

## 30 000-ft view

```mermaid
flowchart LR
    User[Python user code] -->|langgraph.* imports| Shim[python/langgraph facade]
    Shim -->|forwards to| Py[rustakka_langgraph package]
    Py -->|PyO3 cdylib| Native[rustakka_langgraph._native]
    Native --> Coord[GraphCoordinator actor]
    Coord -->|plan/dispatch| NodeTasks[tokio::spawn per node]
    NodeTasks -->|NodeDone| Coord
    Coord -->|StreamEvent::*| Bus[StreamBus]
    Bus -->|mpsc| Native
    Coord -->|put/get_tuple| Saver[Checkpointer]
    Saver --> DB[(Memory / SQLite / Postgres)]
    Coord -->|get/put| Store[BaseStore]
```

Highlights:

- **One coordinator per run.** `GraphCoordinator` is a rustakka `Actor`;
  every `invoke`/`stream` call spawns a fresh coordinator so runs are
  isolated and cancellable without touching shared state.
- **Actors over locks.** Channel ledgers, the pending-task set, and the
  stream subscriber list all live inside the coordinator's private
  state — the actor mailbox serialises all mutations, eliminating the
  need for mutexes on the hot path.
- **GIL-free core.** Everything under `crates/rustakka-langgraph-*` is
  pure Rust. The PyO3 layer acquires the GIL only to (a) call user
  callables and (b) marshal values across the FFI boundary.

## Crate topology

| Crate | Responsibility |
| --- | --- |
| `rustakka-langgraph-core` | Pregel engine: channels, state, graph builder, `GraphCoordinator`, `NodeKind`, runner, streaming bus, visualization. |
| `rustakka-langgraph-checkpoint` | `Checkpointer` trait, `MemorySaver`, and the `CheckpointerHookAdapter` that bridges savers to the coordinator. |
| `rustakka-langgraph-checkpoint-sqlite` | `SqliteSaver` with upstream schema parity. |
| `rustakka-langgraph-checkpoint-postgres` | `PostgresSaver` + `AsyncPostgresSaver` alias, upstream DDL. |
| `rustakka-langgraph-store` | `BaseStore` trait, `InMemoryStore` with TTL + semantic search, `Embedder` trait, and the `StoreAccessor` bridge consumed by the core. |
| `rustakka-langgraph-store-postgres` | `PostgresStore` + `AsyncPostgresStore` alias, upstream DDL. |
| `rustakka-langgraph-providers` | OpenAI / Azure / Vertex / Ollama / Bedrock chat-model clients with a shared `ChatModel` trait, plus a `MockChatModel` for tests. |
| `rustakka-langgraph-prebuilt` | `ToolNode`, `tools_condition`, `create_react_agent`, `create_supervisor`, `create_swarm`. |
| `rustakka-langgraph-macros` | `#[derive(GraphState)]` macro (channel specs + reducers). |
| `rustakka-langgraph` | Umbrella re-export with feature flags (`sqlite`, `postgres`, `prebuilt`, `providers`). |
| `rustakka-langgraph-profiler` | Cross-runtime profiler with `invoke`/`stream`/`fanout`/`checkpoint-heavy` scenarios. |
| `py-bindings/pylanggraph` | PyO3 cdylib exposing `rustakka_langgraph._native`. |

## The Pregel barrier

Every run advances as a sequence of *supersteps*. Each superstep has
three phases that execute strictly in order inside `GraphCoordinator`:

1. **Plan** — compute the next batch of `DispatchTarget`s from the graph
   topology, any outstanding `Command::goto`, and pending `Send`
   fan-outs. Enforce `CompileConfig.recursion_limit` (with
   `RunnableConfig.recursion_limit` as a per-run override) and bail with
   `GraphError::Recursion` when the ceiling is crossed. Static
   breakpoints declared via `CompileConfig.interrupt_before` also fire
   here — the coordinator pauses, persists a checkpoint, and surfaces
   the `Interrupt` to the caller before dispatching the named node.
2. **Execute** — spawn a `tokio::task` per target. The coordinator owns
   a `HashSet<task_id>` (`state.pending`). Each spawned task installs a
   task-local `StreamWriter` (see *Streaming*) and, when a store is
   attached, a `CURRENT_STORE`. Nodes with a `CachePolicy` check the
   per-graph cache and short-circuit on a fresh hit; otherwise the node
   executes through `invoke_with_retry` (honouring `RetryPolicy`) and
   the coordinator receives `CoordMsg::NodeDone { task_id, node, result }`.
3. **Update** — once `pending` drains, aggregate writes through the
   channel reducers, emit `Updates` / `Values` / `OnChainEnd` stream
   events, drain any `Command { graph: "PARENT", .. }` escalation into
   `state.parent_command`, honour `CompileConfig.interrupt_after`, write
   per-task pending writes via `Checkpointer::put_writes`, then flush
   the end-of-step checkpoint according to the configured
   [`Durability`](#durability-modes). Finally the router plans the next
   cycle from the post-update values.

The `task_id` (not the node name) is the dispatch identity. This keeps
`Send`-driven fan-out — where the same node name appears multiple times
in one superstep — race-free.

### Fan-out, Command, Interrupts, Subgraphs

- **`Command`** — a node can return `NodeOutput::Command(Command)` to
  combine explicit `goto`, arbitrary channel `update`s, and outbound
  `Send`s. Commands are interpreted during the Update phase *after*
  writes have been applied so conditional routers see the post-update
  values.
- **`Send`** — each `Send { node, arg }` becomes a fresh
  `DispatchTarget` in the *next* superstep with its `arg` injected as
  `__send_arg__` in the node's input map.
- **Interrupts** — a node returns `NodeOutput::Interrupted(Interrupt)`;
  the coordinator persists the snapshot, surfaces the payload to the
  caller, and pauses. Calling `resume()` rehydrates the channel state
  from the checkpoint and dispatches the interrupted node with the
  resume payload injected via `input_override`.
- **Subgraphs** — `NodeKind::Subgraph(Arc<dyn SubgraphInvoker>)` lets a
  compiled child graph participate as a single node. The parent
  coordinator treats its output as channel writes; the child owns its
  own coordinator and channel state. When the child returns a
  `Command { graph: Some("PARENT"), .. }` the subgraph adapter
  forwards it to the parent's update phase so the child can redirect
  the outer flow (matches upstream's
  `Command(graph=Command.PARENT, goto=...)`). The child's stream events
  are forwarded to the parent bus with the subgraph node name prepended
  to their `namespace`, which is how `stream(subgraphs=True)` surfaces
  nested traffic.

### Resume semantics

`GraphCoordinator::start_run` treats every `StartRun` as follows:

- If a checkpointer is attached, try to load the latest snapshot.
- If the snapshot has a pending interrupt **and** a resume value, route
  straight to the interrupted node.
- Otherwise, always dispatch the graph's entry targets with whatever
  channel values were restored. This lets a finished thread be
  re-invoked to extend its state — a no-op run on a cleared topology
  would otherwise hang forever.

## Channels and state

Channels live in `GraphValues` (inside `state.rs`). Each channel has a
`ChannelSpec { name, reducer }` where `reducer` is one of:

- `last_value` — overwrite on write (default)
- `topic` — append-and-preserve-order
- `topic_unique` — set-like dedupe
- `merge_dicts` — merge `Value::Object`s
- `binary_add_numbers` / `binary_extend_arrays` — numeric / array reducers
- `add_messages` — upstream `BaseMessage` semantics, including removal
  by `remove` markers
- `ephemeral` — cleared at the start of every superstep

The `#[derive(GraphState)]` macro (`rustakka-langgraph-macros`) auto-
generates channel specs from `Annotated[..., reducer]` style type
annotations.

`GraphValues::snapshot()` returns a portable, JSON-serialisable
representation; `restore()` is the inverse. Checkpointers use these
unchanged — byte-for-byte compatible with upstream snapshots.

## Streaming

The `StreamBus` is a lightweight broadcast fan-out (not a rustakka actor
— it's on the hot path and we want minimal indirection). Every event
carries a `namespace: Vec<String>` which is empty for the root graph and
populated with the `[subgraph_node, …]` path when forwarded from a
child.

| Event | Mode | Emitted when |
| --- | --- | --- |
| `Values { step, values, namespace }` | `values` | End of every Update phase. |
| `Updates { step, node, update, namespace }` | `updates` | Per-node reducer output during Update. |
| `Messages { step, node, message, namespace }` | `messages` | Node calls `current_writer().message(..)`. |
| `Custom { step, node, payload, namespace }` | `custom` | Node calls `current_writer().custom(..)`. |
| `Debug { step, payload, namespace }` | `debug` | When `CompileConfig { debug: true, .. }`. |
| `OnChainStart` / `OnChainEnd` | `events` | Before / after every node invocation. |
| `OnChatModelStream` | `events` | Per-chunk LLM token deltas. |
| `OnToolStart` / `OnToolEnd` | `events` | When a prebuilt tool begins / finishes. |

Subscribers register with a list of `StreamMode`s; an empty list means
"all". Under the hood the coordinator installs a per-task
`CURRENT_WRITER` `tokio::task_local!`; nodes pick it up via the
`rustakka_langgraph_core::stream::current_writer()` helper. The v2 event
kinds mirror upstream's `astream_events` so tracing integrations can be
built on the same payload shape.

See [streaming.md](streaming.md) for end-to-end samples in both Rust and
Python.

## Store injection

Nodes reach their attached `BaseStore` through the same task-local
pattern used for streaming. The core crate defines a minimal
`StoreAccessor` trait (`get` / `put` / `delete` / `search`) plus a
`CURRENT_STORE` `tokio::task_local!`; the `rustakka-langgraph-store`
crate provides `store_accessor(Arc<S>)` which adapts any `BaseStore` to
the trait, and the coordinator installs it around every node invocation.

```rust
use rustakka_langgraph_core::context::get_store;

NodeKind::from_fn(|_input| async move {
    if let Some(store) = get_store() {
        store.put(&["prefs".into()], "theme",
                  serde_json::json!("dark"), None).await?;
    }
    Ok(NodeOutput::Update(Default::default()))
});
```

Runs opt in via `runner::invoke_with_store(app, input, cfg, cp, store)`
(or, from Python, `CompiledStateGraph.attach_store(...)` before
invoking). `InMemoryStore::with_embedder(emb, fields)` enables
similarity search: every `put` pipes the configured fields through the
`Embedder` trait and `search(ns, Some(query), …)` ranks the results by
cosine similarity. A `HashingEmbedder` is included for tests /
zero-dependency deployments; real deployments swap in an LLM provider
embedding model.

See [store.md](store.md) for usage, schema, and the embedder contract.

## Node cache

`CachePolicy` (in `graph.rs`) attaches to a node via
`StateGraph::set_cache_policy(name, policy)`. Each compiled graph owns a
`NodeCache` (`parking_lot::RwLock<HashMap<(node, key), CacheEntry>>`);
when a cached node runs, the coordinator computes `policy.key_func(&input)`,
looks up the entry, honours `policy.ttl_seconds`, and — on a miss — runs
the node normally and stores the resulting `NodeOutput::Update` map.
Commands, Sends, and Interrupts are intentionally *not* cached because
they are control-flow payloads that must be replayed live.

## Durability modes

`CompileConfig.durability` selects when the coordinator flushes
checkpoints:

- `Sync` *(default)* — await the saver inside the Update phase. Safest;
  the caller sees `invoke()` return only after every write has landed.
- `Async` — spawn the saver call and continue. Errors surface through
  `tracing`; the run completes without waiting for disk.
- `Exit` — defer every flush until the run finishes, then write a single
  final checkpoint. Useful for batch pipelines where intermediate
  snapshots add no value.

Pending-writes (`Checkpointer::put_writes`) are always persisted
synchronously; only the end-of-step snapshot observes the mode selector.

## Visualization

`CompiledStateGraph::draw_mermaid()` and `draw_ascii()` (in
`crates/rustakka-langgraph-core/src/visualize.rs`) emit deterministic
string renderings suitable for notebooks, logs, or Mermaid-aware
rendering tools. Conditional edges are shown as dotted arrows; branch
labels are preserved when a `path_map` is supplied. Both are exposed as
Python methods on `CompiledStateGraph`.

## Checkpointing

`Checkpointer` (in `rustakka-langgraph-checkpoint::base`) mirrors
upstream's `BaseCheckpointSaver`: `put`, `get_tuple`, `list`,
`put_writes`, plus `setup`. Implementations:

- `MemorySaver` — thread-safe `BTreeMap` keyed by
  `(thread_id, checkpoint_ns, checkpoint_id)`.
- `SqliteSaver` — `sqlx::sqlite` with the upstream DDL
  (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`,
  `checkpoint_migrations`). Pool-backed so concurrent runs don't
  serialise.
- `PostgresSaver` / `AsyncPostgresSaver` — schema-identical; supports
  custom `schema=` for multi-tenant deployments.

The coordinator talks to savers through `CheckpointerHook` — a small
trait exposing `put_step`, `get_latest`, `get_at`, `put_writes`, and
`list_checkpoints`. `get_at` powers time-travel (resume from a specific
`RunnableConfig.checkpoint_id`); `list_checkpoints` powers
`get_state_history`. `CheckpointerHookAdapter` implements the hook for
any `Checkpointer`, so new backends only need to implement the base
trait.

### State inspection API

The runner exposes four APIs that operate purely on the attached
checkpointer (no coordinator spin-up required):

| API | Purpose |
| --- | --- |
| `runner::get_state(app, cfg, cp)` | Fetch the latest snapshot (or the one at `cfg.checkpoint_id`) as a `StateSnapshot { values, step, interrupt, config }`. |
| `runner::get_state_history(app, cfg, cp, limit)` | List prior checkpoints newest-first, each as a `StateSnapshot`. |
| `runner::update_state(app, cfg, cp, patch, as_node)` | Apply a manual write without executing a node; writes are persisted via `put_writes` and a fresh snapshot. |
| `CompiledStateGraph::with_config` *(Python)* | Returns a handle that merges a default `RunnableConfig` into every subsequent call. |

These power upstream's HITL and time-travel tutorials.

Full details, including thread-id layout and restore semantics, in
[checkpointing.md](checkpointing.md).

## Python bridge

`crates/py-bindings/pylanggraph` is a single cdylib compiled to
`rustakka_langgraph._native`. Key components:

- `runtime.rs` — initialises a multi-thread tokio runtime shared
  between pyo3-async-runtimes and every `block_on` site.
- `py_state_graph.rs` / `py_compiled_state_graph.rs` — expose the
  graph builder and the compiled handle. Every call that drives the
  engine wraps its `block_on` in `py.allow_threads(..)` so the user's
  Python nodes can reacquire the GIL from worker threads without
  deadlocking.
- `py_callable_node.rs` — wraps any Python callable as a
  `NodeKind::Python`. Detects coroutines via `inspect.iscoroutine` and
  bridges them with `pyo3_async_runtimes::tokio::into_future`.
- `conversions.rs` — Python ↔ `serde_json::Value` bridge. JSON is the
  wire format on the FFI boundary: the engine never touches the GIL
  during message passing, and checkpoints are interchangeable with
  upstream Python.
- `py_savers.rs` / `py_stores.rs` — PyO3 wrappers over the concrete
  saver / store types, so Python users can pass them to `.compile()`.

The pure-Python packages live in `python/`:

- `python/rustakka_langgraph/` — canonical imports.
- `python/langgraph/` — forwarder package so `from langgraph.graph
  import StateGraph` etc. continue to work unchanged.

See [python.md](python.md) for the compatibility matrix.

## Profiler

`rustakka-langgraph-profiler` produces the same JSON schema as the
rustakka actor profiler so we can merge results side-by-side:

```bash
cargo run -p rustakka-langgraph-profiler --release -- \
  --scenario invoke --iterations 500
```

Scenarios:

| Scenario | What it exercises |
| --- | --- |
| `invoke` | Single two-step graph end-to-end — latency floor. |
| `fanout` | 8 sibling nodes with parallel dispatch — scheduler throughput. |
| `stream` | Full streaming run with `Values` + `Updates` subscribers — event bus + mpsc cost. |
| `checkpoint-heavy` | Large state (256-item arrays) with `MemorySaver` on every step — reducer + serialisation cost. |

Results include `iterations`, `elapsed_ms`, `p50_us`, and `p99_us`.

## Testing

- **Rust**: 67 tests across core (channels, state, coordinator,
  control-flow, streaming, runner, visualization, cache policy,
  retry policy), checkpointers (memory + sqlite), stores (memory +
  injection + semantic search), providers (OpenAI, mock, shared
  types), and prebuilt (ReAct agent, tool node, supervisor, swarm).
- **Python**: 11 pytest-driven tests covering the facade, Memory
  checkpointer resume, prebuilt, and the functional API.
- **CI**: `.github/workflows/ci.yml` runs Rust + Python suites and the
  profiler smoke scenario on every push.

## Extending

- **New reducer** — add a `ChannelSpec` variant and implement it in
  `channel.rs`; add a matching case to the `GraphState` macro.
- **New checkpointer backend** — implement `Checkpointer`. The adapter
  handles the hook plumbing.
- **New node kind** — extend `NodeKind` and teach `clone_node` +
  `invoke` about it (see how `Subgraph` is wired).
- **New streaming mode** — add a variant to `StreamEvent`,
  `StreamMode`, and plumb it through `StreamWriter`.
- **Python exposure** — add a `#[pyclass]` in `pylanggraph` and a
  forwarder in `python/langgraph/…`. Always wrap native `block_on`
  calls in `py.allow_threads(..)`.
- **New prebuilt pattern** — add a module under
  `crates/rustakka-langgraph-prebuilt/src/` that returns a
  `CompiledStateGraph`. `create_supervisor` / `create_swarm` are good
  references for routing conventions. Re-export the helper from
  `lib.rs` and add a matching Python shim in `python/langgraph/prebuilt`
  when user-facing.
- **Custom `Embedder`** — implement the `Embedder` trait in your crate
  and hand an `Arc<dyn Embedder>` to `InMemoryStore::with_embedder(...)`.
  The in-memory adapter recomputes embeddings on every `put`; Postgres
  stores should persist the vector alongside the row.
