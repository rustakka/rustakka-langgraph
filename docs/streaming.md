# Streaming

`rustakka-langgraph` supports every streaming mode the upstream Python
LangGraph ships: `values`, `updates`, `messages`, `custom`, and
`debug`. The underlying transport is a typed broadcast bus inside the
coordinator that fans events out to any number of subscribers without
slowing the Pregel barrier.

## Event model

```rust
// crates/rustakka-langgraph-core/src/stream.rs
pub enum StreamEvent {
    Values   { step: u32, values: BTreeMap<String, Value> },
    Updates  { step: u32, node: String, update: BTreeMap<String, Value> },
    Messages { step: u32, node: String, message: Value },
    Custom   { step: u32, node: String, payload: Value },
    Debug    { step: u32, payload: Value },
}

pub enum StreamMode { Values, Updates, Messages, Custom, Debug }
```

`StreamBus::subscribe(modes)` returns a `tokio::sync::mpsc::Receiver`;
passing an empty `modes` vector subscribes to everything.

## When each event fires

| Event | Fires during | Trigger |
| --- | --- | --- |
| `Values` | End of the Update phase | After reducers merge the superstep's writes. Delivers the full channel snapshot. |
| `Updates` | Update phase | Once per completed node, after its reducer output has been merged. |
| `Messages` | Execute phase | When a node calls `current_writer().message(..)`. |
| `Custom` | Execute phase | When a node calls `current_writer().custom(..)`. |
| `Debug` | Plan / Update | When `CompileConfig { debug: true, .. }`; payload shape is intentionally unstable. |

## Emitting from a node (Rust)

The coordinator installs a task-local `CURRENT_WRITER` before each node
invocation. Nodes reach for it through the `stream::current_writer()`
helper:

```rust
use rustakka_langgraph_core::stream::current_writer;
use serde_json::json;

let node = NodeKind::from_fn(|_| async move {
    if let Some(w) = current_writer() {
        w.custom(json!({"progress": 0.25}));
        w.message(json!({"role": "assistant", "content": "thinking…"}));
    }
    Ok(NodeOutput::Update(Default::default()))
});
```

## Emitting from a node (Python)

`Values` and `Updates` events are emitted automatically for every Python
graph. Custom / Messages emission from inside a Python node — the
upstream `langgraph.config.get_stream_writer()` API — is on the
roadmap: the task-local already exists in Rust, so exposing it through
the cdylib is a small follow-up. Track this in the compatibility matrix
in [python.md](python.md).

In the meantime, you can still consume every stream mode from the
outside via `app.stream(..., stream_mode=["messages", "custom"])`; the
events just won't include payloads emitted from pure-Python nodes.

## Subscribing from Rust

```rust
use rustakka_langgraph_core::runner::stream_run;
use rustakka_langgraph_core::stream::StreamMode;

let mut rx = stream_run(
    app.clone(),
    input,
    cfg,
    vec![StreamMode::Values, StreamMode::Updates],
).await?;

while let Some(event) = rx.recv().await {
    println!("{event:?}");
}
```

`stream_run` returns only once the coordinator has been created and the
subscription is active, so callers never race with early `Values`
events.

## Subscribing from Python

```python
for event in app.stream({"name": "alice"}, stream_mode=["values", "updates"]):
    kind = event["kind"]
    if kind == "values":
        print("snapshot", event["step"], event["values"])
    elif kind == "updates":
        print("update from", event["node"], event["update"])
```

Accepted `stream_mode` values mirror upstream: a single string, a list
of strings, or `None` for *all*.

The sync `stream()` iterator collects events in order; `astream()`
yields them as an `AsyncIterator`.

## Debug mode

Enable by compiling with `debug=True`:

```python
app = g.compile(debug=True)
for event in app.stream({}, stream_mode="debug"):
    print(event["payload"])
```

Debug payloads include plan/execute/update lifecycle markers. The
schema is intentionally not part of the stable API — it's a diagnostic
channel.

## Backpressure and lifetime

- The bus uses bounded mpsc channels per subscriber (default capacity
  64). If a subscriber can't keep up, the oldest unread event is
  dropped and a warning is logged; runs never block on slow consumers.
- Receivers are automatically dropped when the coordinator finishes,
  which terminates the `for`/`async for` loop naturally.
- Multiple subscribers can coexist — the profiler's `stream` scenario
  exercises this with a `Values` + `Updates` pair.

## Profiling streaming throughput

```bash
cargo run -p rustakka-langgraph-profiler --release -- \
  --scenario stream --iterations 1000
```

This scenario spins up a two-node graph, subscribes to both `Values`
and `Updates`, drives the run, and drains the receiver. The reported
`p50_us` / `p99_us` numbers are end-to-end: coordinator plan →
dispatch → reducer → bus send → Python/Rust consumer.

## Testing streaming

- Rust integration tests:
  `cargo test -p rustakka-langgraph-core --test streaming`
- Python tests: `pytest python/tests/test_state_graph.py -k streaming`

Both suites assert the exact shape of each event variant so regressions
in the bus or coordinator ordering surface immediately.
