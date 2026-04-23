//! Task-local execution context injected by the coordinator.
//!
//! The coordinator sets [`CURRENT_STORE`] (alongside [`crate::stream::CURRENT_WRITER`])
//! before dispatching each node so nodes can perform long-term storage
//! operations via [`get_store`] without having to thread a handle through
//! their signatures — matching upstream's `get_store()` Python helper.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::errors::GraphResult;

/// Minimal store contract visible to node bodies. The `rustakka-langgraph-store`
/// crate provides a richer `BaseStore` trait that can be adapted to this
/// accessor via [`adapter_from_base_store`] (see that crate for the full
/// impl; core must stay dependency-free).
#[async_trait]
pub trait StoreAccessor: Send + Sync + 'static {
    async fn get(&self, namespace: &[String], key: &str) -> GraphResult<Option<Value>>;
    async fn put(
        &self,
        namespace: &[String],
        key: &str,
        value: Value,
        ttl_seconds: Option<u64>,
    ) -> GraphResult<()>;
    async fn delete(&self, namespace: &[String], key: &str) -> GraphResult<()>;
    async fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: u32,
        offset: u32,
    ) -> GraphResult<Vec<SearchResult>>;
}

/// Row returned from [`StoreAccessor::search`]. Kept minimal here so the
/// higher-level [`rustakka_langgraph_store::SearchHit`] can convert cheaply.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub namespace: Vec<String>,
    pub key: String,
    pub value: Value,
    pub score: Option<f32>,
}

tokio::task_local! {
    /// Installed by the coordinator around every node invocation when a
    /// store is attached to the compiled graph. Access via [`get_store`].
    pub static CURRENT_STORE: Arc<dyn StoreAccessor>;
}

/// Returns the store attached to the currently-running graph, if any.
/// Only callable from inside a node body.
pub fn get_store() -> Option<Arc<dyn StoreAccessor>> {
    CURRENT_STORE.try_with(|s| s.clone()).ok()
}
