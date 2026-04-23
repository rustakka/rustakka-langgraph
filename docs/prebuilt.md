# Prebuilt agents

The `rustakka-langgraph-prebuilt` crate ships the same canonical
agent factories as upstream LangGraph: a ReAct tool loop, a supervisor
hub-and-spoke, and a peer-to-peer swarm. Everything returns a
`CompiledStateGraph`, so the resulting agents work with the full
streaming / checkpointing / store-injection stack.

## `ToolNode` and `tools_condition`

Wraps a registry of tools as a node. Each tool is an async closure
`(args: Value) -> Value`; the node consumes the latest message's
`tool_calls` field, runs every tool in parallel, and emits the results
as new `messages` updates via the `add_messages` reducer.

```rust
use rustakka_langgraph_prebuilt::{Tool, ToolNode, tools_condition};

let tools = vec![
    Tool::new("add", "sum two numbers", |args| async move {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok(serde_json::json!(a + b))
    }),
];

g.add_node("tools", ToolNode::new(tools).into_node())?;
g.add_conditional_edges("agent", std::sync::Arc::new(tools_condition), None);
```

`tools_condition` returns `["tools"]` when the last message has a
non-empty `tool_calls` array and `[END]` otherwise.

## `create_react_agent`

Assembles `(agent → tools)` cycling until the LLM stops emitting
`tool_calls`:

```rust
use rustakka_langgraph_prebuilt::{create_react_agent, ReactAgentOptions, Tool};

let app = create_react_agent(
    model_fn,           // Arc<dyn Fn(Vec<Value>, Option<String>) -> Fut>
    vec![Tool::new("add", "sum", /* … */)],
    ReactAgentOptions {
        system_prompt: Some("You are a calculator.".into()),
        recursion_limit: Some(10),
    },
).await?;
```

From Python:

```python
from langgraph.prebuilt import create_react_agent

app = create_react_agent(model, [add_tool], recursion_limit=10)
app.invoke({"messages": [{"role": "user", "content": "40+2?"}]})
```

## `create_supervisor`

Hub-and-spoke multi-agent pattern (mirrors `langgraph-supervisor`). A
supervisor node routes to one of `agents`; every agent loops back to
the supervisor. When the supervisor's router returns `"END"` (or
`__end__`) the graph finishes.

```rust
use rustakka_langgraph_prebuilt::{create_supervisor, Agent};

let supervisor = NodeKind::from_fn(/* LLM that emits a "next" hint */);
let router: SupervisorRouter = Arc::new(|state| {
    match state.get("next").and_then(|v| v.as_str()) {
        Some("researcher") => vec!["researcher".into()],
        Some("writer")     => vec!["writer".into()],
        _                  => vec![END.into()],
    }
});
let graph = create_supervisor(
    supervisor, router,
    vec![
        Agent::new("researcher", research_node),
        Agent::new("writer",     writer_node),
    ],
).await?;
```

The supervisor node itself is usually a chat-model call that writes an
agent name into state; the `router` is a small function that reads
that decision out. Any agent can itself be a compiled subgraph —
compose `CompiledStateGraph::as_subgraph_invoker()` into
`NodeKind::Subgraph`.

**Child → parent control-flow.** A subgraph-agent can escalate back to
the parent flow by returning
`Command { graph: Some("PARENT".into()), goto: vec!["supervisor".into()], .. }`
from any of its nodes. This is the upstream
`Command(graph=Command.PARENT, goto=…)` pattern.

## `create_swarm`

Peer-to-peer handoff pattern (mirrors `langgraph-swarm`). Every agent
is reachable from every other agent by writing `next: "<agent>"` into
state. An empty `next` (or `"END"`) finishes the graph.

```rust
use rustakka_langgraph_prebuilt::{create_swarm, Agent};

let graph = create_swarm(
    vec![
        Agent::new("researcher", research_node),
        Agent::new("coder",      coder_node),
        Agent::new("reviewer",   review_node),
    ],
    "researcher",   // default entry agent
).await?;
```

Agents signal their handoff like this:

```rust
let mut out = BTreeMap::new();
out.insert("messages".into(), /* … */);
out.insert("next".into(), json!("coder"));   // hand off
// or:
out.insert("next".into(), json!(""));        // finish
NodeOutput::Update(out)
```

## Python surface

```python
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph_supervisor import create_supervisor
from langgraph_swarm      import create_swarm
```

All four factories return a `CompiledStateGraph` with the standard
`invoke` / `ainvoke` / `stream` / `astream` / `with_config` /
`get_state` / `update_state` / `draw_mermaid` API surface.

## Composing with other subsystems

- **Checkpointing** — pass a saver to `g.compile(checkpointer=…)` on
  the outer graph; each agent turn is persisted.
- **Store injection** — pass `store=…` to make an `InMemoryStore`
  reachable from every agent via
  `rustakka_langgraph_core::context::get_store()`.
- **Streaming** — subscribe to `stream_mode=["updates", "events"]` with
  `subgraphs=True` to observe every sub-agent's progress. Events from
  sub-agents are namespaced with the agent name.
- **Caching** — wrap an expensive agent's node with
  `StateGraph::set_cache_policy(name, CachePolicy::new(Some(600)))` to
  dedupe identical inputs within a 10-minute window.
