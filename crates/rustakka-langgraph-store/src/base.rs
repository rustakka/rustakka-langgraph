//! `BaseStore` — long-term key/value store with namespaces and TTL.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustakka_langgraph_core::errors::GraphResult;

pub type Namespace = Vec<String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub namespace: Namespace,
    pub key: String,
    pub value: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Optional TTL expiry (UTC). None = no expiry.
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
pub struct PutOptions {
    pub ttl_seconds: Option<u64>,
    /// Optional embedding for vector search; concrete stores decide whether to
    /// compute one automatically. Mirrors upstream's `index` argument.
    pub index: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub item: Item,
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct ListNamespacesFilter {
    pub prefix: Option<Namespace>,
    pub max_depth: Option<u32>,
    pub limit: Option<u32>,
}

/// Mirror of `langgraph.store.base.BaseStore`.
#[async_trait]
pub trait BaseStore: Send + Sync + 'static {
    async fn get(&self, namespace: &Namespace, key: &str) -> GraphResult<Option<Item>>;
    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: Value,
        opts: PutOptions,
    ) -> GraphResult<()>;
    async fn delete(&self, namespace: &Namespace, key: &str) -> GraphResult<()>;
    async fn search(
        &self,
        namespace_prefix: &Namespace,
        query: Option<&str>,
        limit: u32,
        offset: u32,
    ) -> GraphResult<Vec<SearchHit>>;
    async fn list_namespaces(&self, filter: ListNamespacesFilter) -> GraphResult<Vec<Namespace>>;
    async fn setup(&self) -> GraphResult<()> {
        Ok(())
    }
}
