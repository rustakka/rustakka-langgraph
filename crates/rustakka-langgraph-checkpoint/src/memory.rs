//! In-memory checkpointer (mirrors `langgraph.checkpoint.memory.MemorySaver`).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use futures::stream::BoxStream;
use parking_lot::RwLock;

use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::errors::GraphResult;

use crate::base::{
    Checkpoint, CheckpointMetadata, CheckpointTuple, Checkpointer, ListFilter, PendingWrite,
};

#[derive(Default)]
pub struct MemorySaver {
    inner: Arc<RwLock<Inner>>,
}

#[derive(Default)]
struct Inner {
    /// (thread_id, checkpoint_ns) -> ordered list of (id, tuple).
    chains: HashMap<(String, String), Vec<(String, CheckpointTuple)>>,
    /// (thread_id, checkpoint_ns, checkpoint_id, task_id) -> writes.
    writes: HashMap<(String, String, String, String), Vec<PendingWrite>>,
}

impl MemorySaver {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(cfg: &RunnableConfig) -> (String, String) {
        let tid = cfg.thread_id().unwrap_or_default().to_string();
        (tid, cfg.checkpoint_ns().to_string())
    }
}

#[async_trait]
impl Checkpointer for MemorySaver {
    async fn put(
        &self,
        cfg: &RunnableConfig,
        ckpt: &Checkpoint,
        meta: &CheckpointMetadata,
        writes: &[PendingWrite],
    ) -> GraphResult<RunnableConfig> {
        let mut inner = self.inner.write();
        let key = Self::key(cfg);
        let chain = inner.chains.entry(key.clone()).or_default();
        let parent_cfg = chain.last().map(|(_, t)| t.config.clone());
        let mut new_cfg = cfg.clone();
        new_cfg.checkpoint_id = Some(ckpt.id.clone());
        let tup = CheckpointTuple {
            config: new_cfg.clone(),
            checkpoint: ckpt.clone(),
            metadata: meta.clone(),
            pending_writes: writes.to_vec(),
            parent_config: parent_cfg,
        };
        chain.push((ckpt.id.clone(), tup));
        Ok(new_cfg)
    }

    async fn get_tuple(&self, cfg: &RunnableConfig) -> GraphResult<Option<CheckpointTuple>> {
        let inner = self.inner.read();
        let key = Self::key(cfg);
        let Some(chain) = inner.chains.get(&key) else { return Ok(None) };
        if let Some(target_id) = &cfg.checkpoint_id {
            return Ok(chain.iter().find(|(id, _)| id == target_id).map(|(_, t)| t.clone()));
        }
        Ok(chain.last().map(|(_, t)| t.clone()))
    }

    fn list<'a>(
        &'a self,
        cfg: &'a RunnableConfig,
        filter: ListFilter,
    ) -> BoxStream<'a, GraphResult<CheckpointTuple>> {
        let inner = self.inner.read();
        let key = Self::key(cfg);
        let mut all: Vec<CheckpointTuple> = inner
            .chains
            .get(&key)
            .map(|c| c.iter().map(|(_, t)| t.clone()).collect())
            .unwrap_or_default();
        all.sort_by(|a, b| b.checkpoint.created_at.cmp(&a.checkpoint.created_at));
        if let Some(before) = &filter.before {
            if let Some(idx) = all.iter().position(|t| t.checkpoint.id == *before) {
                all.truncate(idx);
            }
        }
        if let Some(limit) = filter.limit {
            all.truncate(limit as usize);
        }
        let stream = futures::stream::iter(all.into_iter().map(Ok));
        Box::pin(stream)
    }

    async fn put_writes(
        &self,
        cfg: &RunnableConfig,
        writes: &[PendingWrite],
        task_id: &str,
    ) -> GraphResult<()> {
        let mut inner = self.inner.write();
        let (tid, ns) = Self::key(cfg);
        let id = cfg.checkpoint_id.clone().unwrap_or_default();
        inner
            .writes
            .entry((tid, ns, id, task_id.to_string()))
            .or_default()
            .extend_from_slice(writes);
        Ok(())
    }
}

/// Helper to build a `Checkpoint` row from coordinator-side state.
pub fn make_checkpoint(
    cfg: &RunnableConfig,
    step: u64,
    values: &BTreeMap<String, serde_json::Value>,
    snapshots: &BTreeMap<String, rustakka_langgraph_core::channel::ChannelSnapshot>,
    interrupt: Option<&rustakka_langgraph_core::command::Interrupt>,
) -> Checkpoint {
    Checkpoint {
        id: uuid::Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)).to_string(),
        thread_id: cfg.thread_id().unwrap_or_default().to_string(),
        checkpoint_ns: cfg.checkpoint_ns().to_string(),
        parent_checkpoint_id: cfg.checkpoint_id.clone(),
        step,
        created_at: Utc::now(),
        channel_values: values.clone(),
        channel_snapshots: snapshots.clone(),
        interrupt: interrupt.cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn put_and_get_roundtrip() {
        let saver = MemorySaver::new();
        let cfg = RunnableConfig::with_thread("t1");
        let mut values = BTreeMap::new();
        values.insert("x".into(), json!(1));
        let ckpt = make_checkpoint(&cfg, 1, &values, &BTreeMap::new(), None);
        let meta = CheckpointMetadata { source: "input".into(), step: 1, ..Default::default() };
        let new_cfg = saver.put(&cfg, &ckpt, &meta, &[]).await.unwrap();
        let got = saver.get_tuple(&new_cfg).await.unwrap().unwrap();
        assert_eq!(got.checkpoint.channel_values["x"], json!(1));
    }
}
