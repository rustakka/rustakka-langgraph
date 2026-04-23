# Python guide

`rustakka-langgraph` ships two Python packages:

- **`rustakka_langgraph`** — the canonical package. Imports the PyO3
  cdylib from `rustakka_langgraph._native` and re-exports every public
  symbol.
- **`langgraph`** — a thin forwarder that matches the upstream
  [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph)
  import paths so existing user code keeps working.

Both live in `python/` and are installed together by `maturin develop`.

## Installing for development

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest pytest-asyncio
maturin develop                       # builds cdylib + installs in-place
```

`scripts/dev.sh` rebuilds the cdylib and reruns pytest in one shot so
you can keep the dev loop fast after Rust-side edits.

## Compatibility matrix

| Upstream import | Supported | Notes |
| --- | --- | --- |
| `from langgraph.graph import StateGraph, START, END, CompiledStateGraph` | ✅ | Exposes `invoke`, `ainvoke`, `stream`, `astream`, `with_config`, `get_state`, `get_state_history`, `update_state`, `draw_mermaid`, `draw_ascii`, `resume`. |
| `from langgraph import MemorySaver` | ✅ | Also exported under `langgraph.checkpoint.memory`. |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | ✅ | Requires the cdylib to be built with the `sqlite` feature (on by default). |
| `from langgraph.checkpoint.postgres import PostgresSaver, AsyncPostgresSaver` | ✅ | Requires `postgres` feature. |
| `from langgraph.store.memory import InMemoryStore` | ✅ | TTL + (optional) embedder-backed semantic search + `list_namespaces`. |
| `from langgraph.store.postgres import PostgresStore, AsyncPostgresStore` | ✅ | Requires `postgres` feature. |
| `from langgraph.types import Command, Send, Interrupt, interrupt` | ✅ | `Command(graph="PARENT", ...)` escalates from a subgraph. |
| `from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition` | ✅ | Pure-Python wrappers that emit a native graph. |
| `from langgraph_supervisor import create_supervisor` / `langgraph_swarm import create_swarm` | ✅ | Thin wrappers over the Rust `create_supervisor` / `create_swarm` factories. |
| `from langgraph.func import entrypoint, task` | ✅ | `@entrypoint(checkpointer=..)` compiles a single-node graph. |
| `from langgraph.config import get_store, get_stream_writer` | 🚧 | Planned — surface `CURRENT_WRITER` / store handle as task-locals. Store is already reachable from *Rust* nodes via `rustakka_langgraph_core::context::get_store`. |

## `StateGraph` API

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

g = StateGraph()              # optional state_schema is accepted but not required
g.add_node("a", lambda s: {"x": 1})
g.add_node("b", lambda s: {"x": s.get("x", 0) + 41})
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)

app = g.compile(checkpointer=MemorySaver())

# Synchronous
out = app.invoke({}, {"configurable": {"thread_id": "t1"}})

# Async
await app.ainvoke({}, {"configurable": {"thread_id": "t1"}})

# Streaming (sync variant collects into a list)
for event in app.stream({}, stream_mode="values"):
    print(event)
```

Supported compile kwargs:

- `checkpointer` — `MemorySaver`, `SqliteSaver`, `PostgresSaver`.
- `store` — `InMemoryStore` or `PostgresStore`; the compiled graph
  threads a `StoreAccessor` into every run so nodes can reach it.
- `interrupt_before` / `interrupt_after` — static breakpoint node
  names (single string or list). Mirrors upstream's HITL pattern.
- `recursion_limit` — compile-time ceiling on supersteps. Overridable
  per-run via `config["recursion_limit"]`.
- `durability` — `"sync"` (default), `"async"`, or `"exit"`. See
  [checkpointing.md](checkpointing.md#durability-modes).
- `debug` — flips the coordinator into debug mode (emits
  `StreamEvent::Debug`).

Runtime `config` mirrors upstream:

```python
{"configurable": {"thread_id": "t1", "checkpoint_ns": "demo"},
 "recursion_limit": 25,
 "checkpoint_id": "optional-for-time-travel"}
```

### Time-travel and manual state edits

When a checkpointer is attached, `CompiledStateGraph` exposes the full
upstream inspection API:

```python
cfg = {"configurable": {"thread_id": "t1"}}
app.invoke({"count": 0}, cfg)

snap   = app.get_state(cfg)                 # latest snapshot
hist   = app.get_state_history(cfg, 10)     # newest first, up to 10 entries
app.update_state(cfg, {"count": 42})        # patch without running a node
app.with_config({"recursion_limit": 5})     # new handle w/ merged defaults
```

`snap` is a dict with `values`, `step`, `interrupt`, and `config`.
Setting `config["checkpoint_id"]` on any call rehydrates that specific
snapshot.

### Attaching a store

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
app   = g.compile(checkpointer=MemorySaver(), store=store)
app.invoke({"user_id": "alice"}, cfg)
```

Python nodes currently access the store via the wrapper object they
already hold; Rust nodes use
`rustakka_langgraph_core::context::get_store()` to reach the same
instance. See [store.md](store.md) for `with_embedder(...)` and
semantic search usage.

## Conditional routing

```python
from langgraph.prebuilt import tools_condition

g.add_conditional_edges("agent", tools_condition)
```

`path` may return a string, a list of strings, or any callable the
underlying router understands. If you pass a `path_map`, its values are
recorded as branch labels exactly like upstream.

## Prebuilt helpers

```python
from langgraph.prebuilt import create_react_agent

def model(messages, system_prompt):
    # Replace with a real LLM call; this demo always calls the tool then finishes.
    if not any("tool_calls" in m for m in messages):
        return {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "1", "name": "add", "args": {"a": 40, "b": 2}}],
        }
    return {"role": "assistant", "content": "the answer is 42"}

def add(args):
    return args["a"] + args["b"]
add.name = "add"

app = create_react_agent(model, [add], recursion_limit=10)
print(app.invoke({"messages": [{"role": "user", "content": "40+2?"}]}))
```

`ToolNode` and `tools_condition` are also importable directly if you
want to wire the graph yourself. See [prebuilt.md](prebuilt.md) for the
`create_supervisor` / `create_swarm` multi-agent factories.

## Graph visualization

```python
print(app.draw_mermaid())   # Mermaid `flowchart TD` for notebooks / docs
print(app.draw_ascii())     # minimal text overview for logs
```

Mirrors upstream's `get_graph().draw_mermaid()` / `draw_ascii()`;
conditional edges are shown as dotted arrows with their branch labels
(when a `path_map` was supplied).

## Functional API

```python
from langgraph import MemorySaver
from langgraph.func import entrypoint, task

@task
def add_one(x: int) -> int:
    return x + 1

@entrypoint(checkpointer=MemorySaver())
def counter(state: dict) -> dict:
    return {"count": add_one(state.get("count", 0))}

cfg = {"configurable": {"thread_id": "t1"}}
counter.invoke({}, config=cfg)  # -> {"count": 1}
counter.invoke({}, config=cfg)  # -> {"count": 2}
```

`@task` currently annotates helpers; a future release will route them
as fan-out targets.

## Async nodes

Coroutine nodes work out of the box — the bindings detect them via
`inspect.iscoroutine` and bridge them through
`pyo3_async_runtimes::tokio::into_future`:

```python
async def chat(state):
    await asyncio.sleep(0)  # pretend we called an async LLM
    return {"messages": [{"role": "assistant", "content": "hi"}]}
```

Sync and async nodes can coexist in the same graph; the coordinator
schedules them on the shared multi-thread tokio runtime and reacquires
the GIL only for the actual call.

## Streaming events

Each event emitted to a Python subscriber is a dict with a `kind` tag:

| `kind` | Extra keys |
| --- | --- |
| `values` | `step`, `values` |
| `updates` | `step`, `node`, `update` |
| `messages` | `step`, `node`, `message` |
| `custom` | `step`, `node`, `payload` |
| `debug` | `step`, `payload` |

See [streaming.md](streaming.md) for how to emit custom/message events
from within a node.

## Gotchas

- **Dict state only** — the native engine normalises state to
  `dict[str, Any]` (JSON-serialisable). If you pass a Pydantic model,
  convert it with `.model_dump()` at the boundary.
- **Reserved channel keys** — `__send_arg__` is reserved by the engine
  to inject `Command::send` payloads; don't use it in your schema.
- **`filterwarnings = ["error"]`** is enabled in `pyproject.toml`. Any
  deprecation warning is promoted to a test failure — please fix the
  call site rather than silencing.

## Testing your graphs

```bash
pytest python/tests -v
```

The suite under `python/tests/` doubles as runnable examples for the
above APIs.
