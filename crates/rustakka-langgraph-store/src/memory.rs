//! `InMemoryStore` — `BTreeMap` backend with TTL pruning.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration, Utc};
use parking_lot::RwLock;
use serde_json::Value;

use rustakka_langgraph_core::errors::GraphResult;

use crate::base::{
    BaseStore, Item, ListNamespacesFilter, Namespace, PutOptions, SearchHit,
};
use crate::embedding::{cosine, Embedder};

#[derive(Default)]
pub struct InMemoryStore {
    inner: Arc<RwLock<BTreeMap<(Namespace, String), Item>>>,
    embeddings: Arc<RwLock<BTreeMap<(Namespace, String), Vec<f32>>>>,
    embedder: Option<Arc<dyn Embedder>>,
    /// Paths (JSON pointer-ish `foo.bar`) inside `value` to embed. Empty
    /// means "embed the whole value as a json string".
    index_fields: Vec<String>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable embedding-backed semantic search. When set, [`search`] with a
    /// non-empty `query` ranks hits by cosine similarity; without an
    /// embedder it falls back to JSON substring matching (original
    /// behaviour).
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>, fields: Vec<String>) -> Self {
        self.embedder = Some(embedder);
        self.index_fields = fields;
        self
    }

    fn extract_text(&self, value: &Value) -> String {
        if self.index_fields.is_empty() {
            return value.to_string();
        }
        let mut parts = Vec::with_capacity(self.index_fields.len());
        for f in &self.index_fields {
            let mut cur = value;
            for seg in f.split('.') {
                match cur.get(seg) {
                    Some(v) => cur = v,
                    None => {
                        cur = &Value::Null;
                        break;
                    }
                }
            }
            parts.push(match cur {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            });
        }
        parts.join(" ")
    }

    fn prune_expired(&self) {
        let now = Utc::now();
        let mut g = self.inner.write();
        g.retain(|_, item| match item.expires_at {
            Some(t) => t > now,
            None => true,
        });
    }
}

#[async_trait]
impl BaseStore for InMemoryStore {
    async fn get(&self, namespace: &Namespace, key: &str) -> GraphResult<Option<Item>> {
        self.prune_expired();
        Ok(self.inner.read().get(&(namespace.clone(), key.to_string())).cloned())
    }

    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: Value,
        opts: PutOptions,
    ) -> GraphResult<()> {
        let now = Utc::now();
        let expires_at = opts.ttl_seconds.map(|s| now + Duration::seconds(s as i64));
        // Compute embedding (if enabled) *before* taking the write lock.
        let embedding = if let Some(emb) = &self.embedder {
            // Honor explicit `opts.index` first, else fall back to the
            // store's configured `index_fields`.
            let text = if let Some(paths) = &opts.index {
                if paths.is_empty() {
                    value.to_string()
                } else {
                    let mut parts = Vec::with_capacity(paths.len());
                    for f in paths {
                        let mut cur = &value;
                        for seg in f.split('.') {
                            match cur.get(seg) {
                                Some(v) => cur = v,
                                None => {
                                    cur = &Value::Null;
                                    break;
                                }
                            }
                        }
                        parts.push(match cur {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        });
                    }
                    parts.join(" ")
                }
            } else {
                self.extract_text(&value)
            };
            Some(emb.embed(&text).await?)
        } else {
            None
        };
        let mut g = self.inner.write();
        let item = Item {
            namespace: namespace.clone(),
            key: key.to_string(),
            value,
            created_at: g
                .get(&(namespace.clone(), key.to_string()))
                .map(|i| i.created_at)
                .unwrap_or(now),
            updated_at: now,
            expires_at,
        };
        g.insert((namespace.clone(), key.to_string()), item);
        if let Some(vec) = embedding {
            self.embeddings
                .write()
                .insert((namespace.clone(), key.to_string()), vec);
        }
        Ok(())
    }

    async fn delete(&self, namespace: &Namespace, key: &str) -> GraphResult<()> {
        self.inner.write().remove(&(namespace.clone(), key.to_string()));
        self.embeddings.write().remove(&(namespace.clone(), key.to_string()));
        Ok(())
    }

    async fn search(
        &self,
        namespace_prefix: &Namespace,
        query: Option<&str>,
        limit: u32,
        offset: u32,
    ) -> GraphResult<Vec<SearchHit>> {
        self.prune_expired();

        // Fast path: semantic search when an embedder + query are both set.
        if let (Some(emb), Some(q)) = (self.embedder.as_ref(), query) {
            let qv = emb.embed(q).await?;
            let items = self.inner.read();
            let embs = self.embeddings.read();
            let mut scored: Vec<SearchHit> = items
                .iter()
                .filter(|((ns, _), _)| ns.starts_with(namespace_prefix))
                .map(|((ns, k), item)| {
                    let score = embs
                        .get(&(ns.clone(), k.clone()))
                        .map(|v| cosine(&qv, v))
                        .unwrap_or(0.0);
                    SearchHit { item: item.clone(), score: Some(score) }
                })
                .collect();
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(scored.into_iter().skip(offset as usize).take(limit as usize).collect());
        }

        // Fallback: substring search / recency order.
        let g = self.inner.read();
        let mut hits: Vec<SearchHit> = g
            .iter()
            .filter(|((ns, _), _)| ns.starts_with(namespace_prefix))
            .filter(|(_, item)| {
                match query {
                    None => true,
                    Some(q) => serde_json::to_string(&item.value)
                        .map(|s| s.contains(q))
                        .unwrap_or(false),
                }
            })
            .map(|(_, item)| SearchHit { item: item.clone(), score: None })
            .collect();
        hits.sort_by(|a, b| b.item.updated_at.cmp(&a.item.updated_at));
        Ok(hits.into_iter().skip(offset as usize).take(limit as usize).collect())
    }

    async fn list_namespaces(&self, filter: ListNamespacesFilter) -> GraphResult<Vec<Namespace>> {
        self.prune_expired();
        let g = self.inner.read();
        let mut out: Vec<Namespace> = g
            .keys()
            .map(|(ns, _)| ns.clone())
            .filter(|ns| {
                filter
                    .prefix
                    .as_ref()
                    .map(|p| ns.starts_with(p))
                    .unwrap_or(true)
            })
            .filter(|ns| {
                filter
                    .max_depth
                    .map(|d| (ns.len() as u32) <= d)
                    .unwrap_or(true)
            })
            .collect();
        out.sort();
        out.dedup();
        if let Some(l) = filter.limit {
            out.truncate(l as usize);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn put_get_delete() {
        let s = InMemoryStore::new();
        let ns = vec!["users".into(), "alice".into()];
        s.put(&ns, "prefs", json!({"theme":"dark"}), PutOptions::default())
            .await
            .unwrap();
        let i = s.get(&ns, "prefs").await.unwrap().unwrap();
        assert_eq!(i.value, json!({"theme":"dark"}));
        s.delete(&ns, "prefs").await.unwrap();
        assert!(s.get(&ns, "prefs").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn ttl_expires() {
        let s = InMemoryStore::new();
        let ns = vec!["x".into()];
        s.put(&ns, "k", json!(1), PutOptions { ttl_seconds: Some(0), ..Default::default() })
            .await
            .unwrap();
        // give the clock a tick
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        assert!(s.get(&ns, "k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn semantic_search_ranks_by_cosine_similarity() {
        use crate::embedding::HashingEmbedder;
        let emb: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(32));
        let s = InMemoryStore::new()
            .with_embedder(emb, vec!["text".into()]);
        let ns = vec!["docs".into()];
        s.put(&ns, "a", json!({"text":"cats are great pets"}), PutOptions::default())
            .await
            .unwrap();
        s.put(&ns, "b", json!({"text":"dogs bark loudly"}), PutOptions::default())
            .await
            .unwrap();
        s.put(&ns, "c", json!({"text":"cats climb trees"}), PutOptions::default())
            .await
            .unwrap();

        let hits = s.search(&ns, Some("cats trees"), 10, 0).await.unwrap();
        assert_eq!(hits.len(), 3);
        // `c` contains both `cats` and `trees` tokens and should lead.
        assert_eq!(hits[0].item.key, "c");
        assert!(hits[0].score.unwrap() >= hits[1].score.unwrap());
    }

    #[tokio::test]
    async fn search_and_list_namespaces() {
        let s = InMemoryStore::new();
        let a = vec!["users".into(), "alice".into()];
        let b = vec!["users".into(), "bob".into()];
        let c = vec!["rooms".into(), "42".into()];
        s.put(&a, "prefs", json!({"theme":"dark"}), PutOptions::default()).await.unwrap();
        s.put(&b, "prefs", json!({"theme":"light"}), PutOptions::default()).await.unwrap();
        s.put(&c, "info", json!({"name":"lobby"}), PutOptions::default()).await.unwrap();

        let hits = s.search(&vec!["users".into()], None, 10, 0).await.unwrap();
        assert_eq!(hits.len(), 2);

        let filtered = s.search(&vec!["users".into()], Some("dark"), 10, 0).await.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].item.namespace, a);

        let nss = s
            .list_namespaces(ListNamespacesFilter {
                prefix: Some(vec!["users".into()]),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(nss.contains(&a));
        assert!(nss.contains(&b));
        assert!(!nss.contains(&c));
    }
}
