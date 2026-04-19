# rustakka-langgraph

An idiomatic, native-Rust port of
[LangGraph](https://github.com/langchain-ai/langgraph) built on the
[rustakka](https://github.com/cognect/rustakka) Akka-style actor runtime,
with a `pyo3`-powered Python shim that keeps the upstream `langgraph` API
drop-in compatible.

All nine plan phases (0–8) are implemented. See
[docs/TODO.md](docs/TODO.md) for the checklist and
[docs/architecture.md](docs/architecture.md) for a deep-dive.

## Why

LangGraph models agentic workflows as cyclic Pregel/BSP graphs. The
pure-Python implementation is bottlenecked by the GIL, deep-copy
checkpointing, and single-threaded `asyncio` orchestration. By moving the
engine to Rust on top of rustakka actors we get:

- **True OS-level parallelism** for planning, dispatch, and node fan-out.
- **Zero-copy channel snapshots** inside the Pregel barrier, with
  `serde` / `rmp-serde` checkpoints that share the upstream SQLite and
  Postgres schemas so existing databases interchange.
- **Strict GIL isolation** — the engine runs GIL-free; Python is
  reacquired only to invoke user callables and marshal stream events
  across the FFI boundary.
- **Drop-in Python compatibility** — `from langgraph.graph import
  StateGraph` continues to work unchanged.

## Workspace layout

```
crates/
  rustakka-langgraph-core/                Pregel engine + channels + state
  rustakka-langgraph-checkpoint/          Checkpointer trait + MemorySaver
  rustakka-langgraph-checkpoint-sqlite/   sqlx SqliteSaver (schema parity)
  rustakka-langgraph-checkpoint-postgres/ sqlx PostgresSaver + AsyncPostgresSaver
  rustakka-langgraph-store/               BaseStore trait + InMemoryStore
  rustakka-langgraph-store-postgres/      PostgresStore + AsyncPostgresStore
  rustakka-langgraph-prebuilt/            create_react_agent, ToolNode, tools_condition
  rustakka-langgraph-macros/              #[derive(GraphState)], #[node]
  rustakka-langgraph/                     Umbrella facade (feature-gated)
  rustakka-langgraph-profiler/            Cross-runtime profiler scenarios
  py-bindings/pylanggraph/                PyO3 cdylib -> rustakka_langgraph._native
python/
  rustakka_langgraph/                     Python package wrapping the cdylib
  langgraph/                              Pure-python facade matching upstream
  tests/                                  pytest parity suite
examples/                                 Pure-Rust graph examples
docs/                                     Architecture, status, guides
scripts/                                  dev-env / dev-loop tooling
```

## Quick start (Rust)

```rust
use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use rustakka_langgraph::core::config::RunnableConfig;
use rustakka_langgraph::core::graph::{CompileConfig, StateGraph, END, START};
use rustakka_langgraph::core::node::{NodeKind, NodeOutput};
use rustakka_langgraph::core::runner::invoke_dynamic;
use rustakka_langgraph::core::state::DynamicState;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let mut g = StateGraph::<DynamicState>::new();
    g.add_node(
        "greet",
        NodeKind::from_fn(|input: BTreeMap<String, serde_json::Value>| async move {
            let name = input.get("name").and_then(|v| v.as_str()).unwrap_or("world");
            let mut m = BTreeMap::new();
            m.insert("out".into(), json!(format!("hi {name}")));
            Ok(NodeOutput::Update(m))
        }),
    )?;
    g.add_edge(START, "greet");
    g.add_edge("greet", END);

    let app = Arc::new(g.compile(CompileConfig::default()).await?);
    let mut input = BTreeMap::new();
    input.insert("name".into(), json!("alice"));
    let out = invoke_dynamic(app, input, RunnableConfig::default()).await?;
    println!("{out:#?}");
    Ok(())
}
```

## Quick start (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest pytest-asyncio
maturin develop
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

g = StateGraph()
g.add_node("hello", lambda s: {"out": f"hi {s.get('name', 'world')}"})
g.add_edge(START, "hello")
g.add_edge("hello", END)

app = g.compile(checkpointer=MemorySaver())
print(app.invoke({"name": "alice"}, {"configurable": {"thread_id": "t1"}}))
```

### Prebuilt ReAct agent

```python
from langgraph.prebuilt import create_react_agent

def model(messages, system_prompt):
    # real code would call an LLM here
    return {"role": "assistant", "content": "done"}

def add(args):
    return args["a"] + args["b"]
add.name = "add"

app = create_react_agent(model, [add], recursion_limit=10)
app.invoke({"messages": [{"role": "user", "content": "hi"}]})
```

### Functional API

```python
from langgraph.func import entrypoint, task

@task
def square(x: int) -> int:
    return x * x

@entrypoint()
def workflow(state: dict) -> dict:
    return {"x": square(state.get("x", 0))}

workflow({"x": 6})  # -> {"x": 36}
```

## Building & testing

```bash
# Rust: 30 tests across 10 crates
cargo build --workspace
cargo test  --workspace

# Python: 11 tests (rebuild the cdylib first)
source .venv/bin/activate
maturin develop
pytest python/tests -v

# Profiler scenarios (release build)
cargo run -p rustakka-langgraph-profiler --release -- \
  --scenario invoke --iterations 100
# Scenarios: invoke | fanout | stream | checkpoint-heavy
```

## Environment configuration

`scripts/dev-env.sh` activates `.venv` and exports a sane
`PYO3_CONFIG_FILE` if you don't have `python3-dev` installed. Runtime
configuration follows the standard 12-factor pattern; look for
`RUSTAKKA_LANGGRAPH_ENV ∈ {dev,test,prod}`.

## Docs

- [docs/architecture.md](docs/architecture.md) — engine, coordinator
  barrier, channels, GIL discipline
- [docs/python.md](docs/python.md) — Python facade + compatibility notes
- [docs/checkpointing.md](docs/checkpointing.md) — Memory / SQLite /
  Postgres savers and schema parity
- [docs/streaming.md](docs/streaming.md) — Values / Updates / Messages /
  Custom / Debug streaming modes
- [docs/TODO.md](docs/TODO.md) — phase-by-phase implementation status
- [resources/Rust LangGraph with Python Bridge.md](resources/Rust%20LangGraph%20with%20Python%20Bridge.md)
  — original architectural specification
