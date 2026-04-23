# Streaming

`rustakka-langgraph` supports every streaming mode the upstream Python
LangGraph ships — `values`, `updates`, `messages`, `custom`, `debug`,
plus the newer `events` mode that powers `astream_events()` v2. The
underlying transport is a typed broadcast bus inside the coordinator
that fans events out to any number of subscribers without slowing the
Pregel barrier. Events from subgraphs carry a `namespace: Vec<String>`
so callers can distinguish nested graphs when `subgraphs=True`.

## Event model

```rust
// crates/rustakka-langgraph-core/src/stream.rs
pub enum StreamEvent {
    Values   { step: u64, values: BTreeMap<String, Value>, namespace: Vec<String> },
    Updates  { step: u64, node: String, update: BTreeMap<String, Value>, namespace: Vec<String> },
    Messages { step: u64, node: String, message: Value, namespace: Vec<String> },
    Custom   { step: u64, node: String, payload: Value, namespace: Vec<String> },
    Debug    { step: u64, payload: Value, namespace: Vec<String> },

    // astream_events v2 ------------------------------------------
    OnChainStart      { step: u64, node: String, run_id: Option<String>, tags: Vec<String>, namespace: Vec<String> },
    OnChainEnd        { step: u64, node: String, run_id: Option<String>, output: Value, namespace: Vec<String> },
    OnChatModelStream { step: u64, node: String, chunk: Value, namespace: Vec<String> },
    OnToolStart       { step: u64, node: String, tool: String, input: Value, namespace: Vec<String> },
    OnToolEnd         { step: u64, node: String, tool: String, output: Value, namespace: Vec<String> },
}

pub enum StreamMode { Values, Updates, Messages, Custom, Debug, Events }
```

`StreamBus::subscribe(modes)` returns a `tokio::sync::mpsc::Receiver`;
passing an empty `modes` vector subscribes to everything. All
`astream_events` v2 variants live under `StreamMode::Events`.

## When each event fires

| Event | Fires during | Trigger |
| --- | --- | --- |
| `Values` | End of the Update phase | After reducers merge the superstep's writes. Delivers the full channel snapshot. |
| `Updates` | Update phase | Once per completed node, after its reducer output has been merged. |
| `Messages` | Execute phase | When a node calls `current_writer().message(..)`. |
| `Custom` | Execute phase | When a node calls `current_writer().custom(..)`. |
| `Debug` | Plan / Update | When `CompileConfig { debug: true, .. }`; payload shape is intentionally unstable. |
| `OnChainStart` / `OnChainEnd` | Execute / Update | Immediately before dispatch and after a node's writes are merged. |
| `OnChatModelStream` | Execute | When a node calls `current_writer().chat_model_chunk(..)`. |
| `OnToolStart` / `OnToolEnd` | Execute | `ToolNode` (and user code) emits these around each tool invocation. |

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
        // astream_events v2 helpers
        w.chat_model_chunk(json!({"content": "hi"}));
        w.tool_start("calc", json!({"a": 1, "b": 2}));
        w.tool_end("calc", json!(3));
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
    mode, payload = event  # multi-mode subscribers always receive (mode, event)
    if mode == "values":
        print("snapshot", payload["step"], payload["values"])
    elif mode == "updates":
        print("update from", payload["node"], payload["update"])
```

Accepted `stream_mode` values mirror upstream: a single string, a list
of strings, or `None` for *all*. The shape of each yielded element
depends on which options are set:

| `stream_mode` | `subgraphs` | Yield shape |
| --- | --- | --- |
| single | `False` (default) | bare event dict |
| list | `False` | `(mode, event)` |
| single | `True` | `(namespace_tuple, event)` |
| list | `True` | `(namespace_tuple, mode, event)` |

Subscribing to `events` returns the v2 kinds — each payload carries an
`event` field (`"on_chain_start"`, `"on_tool_end"`, …) plus the fields
enumerated in the event model table.

The sync `stream()` iterator collects events in order; `astream()`
yields them as an `AsyncIterator`.

## Subgraph namespacing

When a subgraph runs as a node inside a parent graph, its stream events
are forwarded onto the parent bus with the subgraph node name prepended
to their `namespace`:

```python
for ns, mode, event in app.stream({}, stream_mode=["updates"], subgraphs=True):
    # ns == ("planner",) for events emitted inside the `planner` subgraph
    # ns == ()            for events emitted by the root graph
    ...
```

This lets a single subscriber observe the full nested trace without
juggling multiple buses.

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
