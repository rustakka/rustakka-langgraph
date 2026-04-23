# Long-term stores

`BaseStore` is the long-term, cross-thread key/value store that sits
next to the per-thread `Checkpointer`. It holds the durable memory of
an agent — user preferences, episodic recall, RAG chunks, any state
that should survive individual graph runs. All backends implement the
same trait and share the upstream LangGraph schema, so
`rustakka-langgraph` can consume databases written by the Python
reference implementation and vice versa.

## The `BaseStore` trait

```rust
// crates/rustakka-langgraph-store/src/base.rs
#[async_trait]
pub trait BaseStore: Send + Sync + 'static {
    async fn put(&self, namespace: &Namespace, key: &str,
                 value: Value, opts: PutOptions) -> GraphResult<()>;
    async fn get(&self, namespace: &Namespace, key: &str)
                -> GraphResult<Option<Item>>;
    async fn delete(&self, namespace: &Namespace, key: &str)
                -> GraphResult<()>;
    async fn search(&self, prefix: &Namespace, query: Option<&str>,
                    limit: u32, offset: u32) -> GraphResult<Vec<SearchHit>>;
    async fn list_namespaces(&self, filter: ListNamespacesFilter)
                -> GraphResult<Vec<Namespace>>;
}
```

`Namespace = Vec<String>` (e.g. `["users", "alice", "prefs"]`).
`PutOptions { ttl_seconds, index, .. }` carries TTL + a per-item index
override. `SearchHit { item, score }` carries an optional similarity
score when the backend supports vector search.

## Supported backends

| Backend | Crate | Notes |
| --- | --- | --- |
| `InMemoryStore` | `rustakka-langgraph-store` | `BTreeMap` keyed by `(namespace, key)` with TTL and optional cosine-similarity search. |
| `PostgresStore` / `AsyncPostgresStore` | `rustakka-langgraph-store-postgres` | Schema-identical to upstream; accepts a custom `schema=` for multi-tenant setups. |

Both expose Python wrappers through `pylanggraph` and are re-exported
from `langgraph.store.{memory,postgres}`.

## Store injection into nodes

The core crate defines a minimal `StoreAccessor` trait (`get` / `put` /
`delete` / `search`) and a `CURRENT_STORE` `tokio::task_local!`. The
`rustakka-langgraph-store` crate provides an adapter:

```rust
use rustakka_langgraph_store::{store_accessor, InMemoryStore};

let store  = store_accessor(Arc::new(InMemoryStore::new()));
let result = rustakka_langgraph_core::runner::invoke_with_store(
    app, input, cfg, /* checkpointer */ None, store,
).await?;
```

Inside a node:

```rust
use rustakka_langgraph_core::context::get_store;

NodeKind::from_fn(|_input| async move {
    if let Some(store) = get_store() {
        store.put(&["prefs".into()], "theme",
                  serde_json::json!("dark"), None).await?;
        let hits = store.search(&["prefs".into()], None, 10, 0).await?;
        tracing::debug!(?hits);
    }
    Ok(NodeOutput::Update(Default::default()))
});
```

From Python, passing `store=` to `g.compile(...)` or calling
`app.attach_store(store)` makes the same accessor available to every
subsequent `invoke` / `stream` call.

## Semantic search

`InMemoryStore::with_embedder(embedder, index_fields)` turns on vector
search. Every `put` pipes the configured fields (dotted JSON paths into
`value`) through the `Embedder` trait and stores the resulting vector
alongside the row. `search(prefix, Some(query), limit, offset)` then
embeds the query and returns rows ranked by cosine similarity. When no
embedder is attached, `search` falls back to substring matching + most
recently updated ordering.

```rust
use rustakka_langgraph_store::{Embedder, HashingEmbedder, InMemoryStore};

let embedder: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(512));
let store = InMemoryStore::new()
    .with_embedder(embedder, vec!["text".into()]);

store.put(&vec!["docs".into()], "a",
          serde_json::json!({"text": "cats climb trees"}),
          Default::default()).await?;
let hits = store.search(&vec!["docs".into()], Some("cats trees"), 5, 0).await?;
```

`HashingEmbedder` is a deterministic, zero-dependency embedder suitable
for tests. Real deployments implement `Embedder` over an LLM provider:

```rust
use async_trait::async_trait;
use rustakka_langgraph_store::Embedder;

struct OpenAiEmbedder { /* … */ }

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn dims(&self) -> usize { 1536 }
    async fn embed(&self, text: &str) -> GraphResult<Vec<f32>> {
        /* call provider */
    }
}
```

`PutOptions::index = Some(vec!["summary".into()])` overrides the store-
wide `index_fields` on a per-put basis, useful when a single store
holds heterogeneous namespaces.

## TTL

`PutOptions::ttl_seconds = Some(3600)` causes the item to disappear
from subsequent `get` / `search` calls after an hour. Expiry is
evaluated lazily inside `search`; `get` returns `None` for an expired
item.

## Python

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(["users", "alice"], "theme", "dark", ttl_seconds=3600)
print(store.get(["users", "alice"], "theme"))       # -> {"value": "dark", ...}
print(store.search(["users", "alice"], query="the")) # substring match
```

Attaching the store to a compiled graph:

```python
app = g.compile(checkpointer=MemorySaver(), store=store)
```

Rust nodes in that graph reach the store via `get_store()`; Python
nodes currently use the wrapper directly. An upstream-style
`get_store()` task-local for Python is on the roadmap.

## Writing a custom backend

Implement `BaseStore` for your type, then use `store_accessor(Arc<S>)`
to bridge it to `StoreAccessor`. Minimum checklist:

- [ ] `put` upserts `(namespace, key, value, updated_at)` atomically.
- [ ] `get` honours TTL (return `None` for expired rows).
- [ ] `search` supports both substring fallback and vector scoring
      when an embedder is attached.
- [ ] `list_namespaces` accepts a prefix + `max_depth` filter.
- [ ] Expose a Python wrapper by adding a `#[pyclass]` and a matching
      `extract::<PyYourStore>()` branch in
      `PyCompiledStateGraph::attach_store`.
