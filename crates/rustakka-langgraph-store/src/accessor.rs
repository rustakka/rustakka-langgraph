//! Bridge: any `BaseStore` → `rustakka_langgraph_core::context::StoreAccessor`.
//!
//! The core engine defines a small `StoreAccessor` contract (so it can stay
//! dep-free). This module provides a zero-cost adapter that lets user code
//! attach any `BaseStore` implementation to a run.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rustakka_langgraph_core::context::{SearchResult, StoreAccessor};
use rustakka_langgraph_core::errors::GraphResult;

use crate::base::{BaseStore, PutOptions};

struct Adapter<S: BaseStore> {
    inner: Arc<S>,
}

#[async_trait]
impl<S: BaseStore> StoreAccessor for Adapter<S> {
    async fn get(&self, namespace: &[String], key: &str) -> GraphResult<Option<Value>> {
        Ok(self.inner.get(&namespace.to_vec(), key).await?.map(|item| item.value))
    }

    async fn put(
        &self,
        namespace: &[String],
        key: &str,
        value: Value,
        ttl_seconds: Option<u64>,
    ) -> GraphResult<()> {
        let opts = PutOptions { ttl_seconds, ..Default::default() };
        self.inner.put(&namespace.to_vec(), key, value, opts).await
    }

    async fn delete(&self, namespace: &[String], key: &str) -> GraphResult<()> {
        self.inner.delete(&namespace.to_vec(), key).await
    }

    async fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: u32,
        offset: u32,
    ) -> GraphResult<Vec<SearchResult>> {
        let hits = self
            .inner
            .search(&namespace_prefix.to_vec(), query, limit, offset)
            .await?;
        Ok(hits
            .into_iter()
            .map(|h| SearchResult {
                namespace: h.item.namespace,
                key: h.item.key,
                value: h.item.value,
                score: h.score,
            })
            .collect())
    }
}

/// Wrap any `Arc<S>` where `S: BaseStore` as a `dyn StoreAccessor` suitable
/// for attaching to a `CompiledStateGraph` run.
pub fn store_accessor<S: BaseStore>(store: Arc<S>) -> Arc<dyn StoreAccessor> {
    Arc::new(Adapter { inner: store })
}
