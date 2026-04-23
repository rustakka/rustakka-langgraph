# Checkpointing

Checkpointers snapshot graph state at the end of every Pregel superstep,
persist any pending writes, and surface the latest snapshot back to
`GraphCoordinator` on the next run. This is the foundation for
time-travel debugging, crash recovery, human-in-the-loop interrupts,
and multi-turn conversational state.

All backends implement the same trait and share the upstream LangGraph
schema, so `rustakka-langgraph` can consume databases written by the
Python reference implementation and vice versa.

## The `Checkpointer` trait

```rust
// crates/rustakka-langgraph-checkpoint/src/base.rs
#[async_trait]
pub trait Checkpointer: Send + Sync + 'static {
    async fn setup(&self) -> GraphResult<()> { Ok(()) }

    async fn put(
        &self,
        cfg: &RunnableConfig,
        ckpt: &Checkpoint,
        meta: &CheckpointMetadata,
        writes: &[PendingWrite],
    ) -> GraphResult<RunnableConfig>;

    async fn get_tuple(&self, cfg: &RunnableConfig)
        -> GraphResult<Option<CheckpointTuple>>;

    fn list<'a>(
        &'a self,
        cfg: &'a RunnableConfig,
        filter: ListFilter,
    ) -> BoxStream<'a, GraphResult<CheckpointTuple>>;

    async fn put_writes(
        &self,
        cfg: &RunnableConfig,
        writes: &[PendingWrite],
        task_id: &str,
    ) -> GraphResult<()>;
}
```

The coordinator drives checkpointers via `CheckpointerHook`, a smaller
trait focused on the hot path. `CheckpointerHookAdapter<C>` implements
the hook for any `Checkpointer`, so new backends only need the base
trait.

## Supported backends

| Backend | Crate | Notes |
| --- | --- | --- |
| `MemorySaver` | `rustakka-langgraph-checkpoint` | `BTreeMap` keyed by `(thread_id, checkpoint_ns, checkpoint_id)`. Thread-safe via `parking_lot`. |
| `SqliteSaver` | `rustakka-langgraph-checkpoint-sqlite` | `sqlx::sqlite` pool, `sqlite::memory:` supported for tests. |
| `PostgresSaver` / `AsyncPostgresSaver` | `rustakka-langgraph-checkpoint-postgres` | Schema-identical to upstream; accepts a custom `schema=` for multi-tenant setups. `AsyncPostgresSaver` is an alias retained for API parity. |

All three expose Python wrappers through `pylanggraph` and are re-
exported from `langgraph.checkpoint.{memory,sqlite,postgres}`.

## Configuration

`RunnableConfig` carries the thread identity through the system:

```rust
pub struct RunnableConfig {
    pub configurable: BTreeMap<String, Value>, // includes thread_id, checkpoint_ns
    pub recursion_limit: Option<u32>,
    pub checkpoint_id: Option<String>,         // point-in-time resume
    // ...
}
```

Helpers:

```rust
RunnableConfig::with_thread("t1")              // fresh thread
rc.thread_id()                                  // Option<&str>
rc.checkpoint_ns()                              // "" by default
```

From Python:

```python
cfg = {"configurable": {"thread_id": "t1", "checkpoint_ns": "demo"},
       "recursion_limit": 25}
app.invoke({}, cfg)
```

## How the coordinator uses the saver

On each `StartRun`:

1. If a saver is attached, call `get_latest(cfg)`. If it returns a
   snapshot:
   - Restore channel values with `GraphValues::restore`.
   - Carry over the pending `Interrupt`, if any.
   - Set `start_step` to the saved step.
2. If there's a pending interrupt *and* a resume value, dispatch the
   interrupted node with the resume injected as `__send_arg__`.
3. Otherwise, always dispatch the graph's entry targets with the
   (possibly restored) values. This is what makes a completed thread
   accept another invocation to extend its state.

After every completed node, the coordinator calls
`Checkpointer::put_writes` with that task's writes — so an in-flight
fan-out step can be replayed on crash without re-running tasks that
already succeeded.

At the end of every Update phase, `put_step` writes the new snapshot,
including every reducer's post-write value plus any pending writes that
came from `Command::update` or `Send`.

### Durability modes

`CompileConfig.durability` (mirrors upstream's
`compile(durability="sync"|"async"|"exit")`) controls when the snapshot
is flushed:

| Mode | Behaviour |
| --- | --- |
| `Sync` (default) | Await the saver inside the Update phase. `invoke()` returns only after the snapshot is durable. |
| `Async` | Spawn the saver call; the run continues immediately. Errors surface via `tracing::warn!`. |
| `Exit` | Defer every snapshot until the run completes, then write a single final checkpoint. |

`put_writes` always runs synchronously — only the end-of-step snapshot
observes the mode. From Python:

```python
app = g.compile(checkpointer=MemorySaver(), durability="async")
```

### Time-travel and state inspection

`runner::get_state(app, cfg, cp)` returns the latest snapshot (or the
one at `cfg.checkpoint_id`) as a `StateSnapshot { values, step,
interrupt, config }`. `runner::get_state_history(app, cfg, cp, limit)`
lists prior checkpoints, newest-first. `runner::update_state(app, cfg,
cp, patch, as_node)` applies a manual write without running a node —
the coordinator persists the patch via `put_writes` and writes a fresh
snapshot so the next `invoke` sees the updated state.

From Python these appear as
`CompiledStateGraph.get_state(config=…)`,
`.get_state_history(config=…, limit=…)`, and
`.update_state(config, values, as_node=None)`. `with_config(config)`
returns a new handle whose `configurable` / `recursion_limit` are
merged into every subsequent call.

## Schema parity

The SQLite and Postgres backends use the **exact DDL shipped by
upstream** (`langgraph-checkpoint-sqlite` and
`langgraph-checkpoint-postgres`):

```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB NOT NULL,
    metadata BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
```

Checkpoint and metadata blobs are stored as JSON (`type = 'json'`) for
byte-for-byte interchange with the Python implementation.

## Rust example

```rust
use std::sync::Arc;
use rustakka_langgraph::checkpoint_sqlite::SqliteSaver;
use rustakka_langgraph_checkpoint::CheckpointerHookAdapter;
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::runner::invoke_with_checkpointer;

let saver = Arc::new(SqliteSaver::from_url("sqlite://./graph.db").await?);
let hook = CheckpointerHookAdapter::new(saver);
let cfg = RunnableConfig::with_thread("t1");
let r1 = invoke_with_checkpointer(app.clone(), input.clone(), cfg.clone(), hook.clone()).await?;
// second call resumes automatically
let r2 = invoke_with_checkpointer(app, input, cfg, hook).await?;
```

## Python example

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

saver = SqliteSaver("sqlite://./graph.db")
app = g.compile(checkpointer=saver)

cfg = {"configurable": {"thread_id": "t1"}}
app.invoke({"count": 0}, cfg)   # writes checkpoint at step 1
app.invoke({}, cfg)              # resumes from step 1, advances to step 2
```

## Interrupts and resume

Within a node, raise an interrupt to pause the graph:

```python
from langgraph import interrupt

def review(state):
    return interrupt({"needs_review": state})
```

The coordinator persists a snapshot that includes the `Interrupt`
payload and surfaces `GraphError::NodeInterrupt` back to the caller.
Resume by invoking with a resume value:

```python
app.resume(cfg, resume_value={"approved": True})
```

The Rust runtime's equivalent is `rustakka_langgraph_core::runner::resume`.

## Testing backends

- In-process: `cargo test -p rustakka-langgraph-checkpoint-sqlite` uses
  `sqlite::memory:` so no files are produced.
- Postgres: point the `POSTGRES_URL` env var at a running instance, or
  use a `testcontainers`-style wrapper in your application's test
  harness. All queries are parameterised and avoid backend-specific
  types beyond `JSONB` + `TIMESTAMPTZ`.

## Writing a custom backend

Implement `Checkpointer` for your type; the
`CheckpointerHookAdapter<C>` takes care of the coordinator-facing
`CheckpointerHook`. Expose a Python wrapper by implementing
`ExtractCheckpointer` (see `py_savers.rs` for an example).

Minimum checklist:

- [ ] `setup` creates tables idempotently.
- [ ] `put` writes `checkpoint` + `metadata` and all `PendingWrite`s
      inside a single transaction.
- [ ] `get_tuple` honours `cfg.checkpoint_id` when set (point-in-time
      restore) and otherwise returns the latest row.
- [ ] `list` orders newest-first and respects `ListFilter::before` /
      `limit`.
- [ ] `put_writes` dedupes by `(thread_id, ns, checkpoint_id, task_id,
      idx)` so retries are idempotent.
