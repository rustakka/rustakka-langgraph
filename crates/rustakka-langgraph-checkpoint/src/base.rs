//! `Checkpointer` trait — Rust mirror of `BaseCheckpointSaver`.

use std::collections::BTreeMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustakka_langgraph_core::channel::ChannelSnapshot;
use rustakka_langgraph_core::command::Interrupt;
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::errors::GraphResult;

/// One row in the `checkpoints` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub thread_id: String,
    pub checkpoint_ns: String,
    pub parent_checkpoint_id: Option<String>,
    pub step: u64,
    pub created_at: DateTime<Utc>,
    /// Materialized `(channel_name -> latest value)` map.
    pub channel_values: BTreeMap<String, Value>,
    /// Per-channel snapshot (for full restore).
    pub channel_snapshots: BTreeMap<String, ChannelSnapshot>,
    /// Optional pending interrupt at the moment this checkpoint was written.
    pub interrupt: Option<Interrupt>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub source: String,
    pub step: u64,
    #[serde(default)]
    pub writes: BTreeMap<String, BTreeMap<String, Value>>,
    #[serde(default)]
    pub parents: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingWrite {
    pub task_id: String,
    pub channel: String,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct CheckpointTuple {
    pub config: RunnableConfig,
    pub checkpoint: Checkpoint,
    pub metadata: CheckpointMetadata,
    pub pending_writes: Vec<PendingWrite>,
    pub parent_config: Option<RunnableConfig>,
}

#[derive(Debug, Clone, Default)]
pub struct ListFilter {
    pub before: Option<String>,
    pub limit: Option<u32>,
    pub metadata: BTreeMap<String, Value>,
}

/// The trait every saver implements. Mirrors upstream's async surface.
#[async_trait]
pub trait Checkpointer: Send + Sync + 'static {
    async fn put(
        &self,
        cfg: &RunnableConfig,
        ckpt: &Checkpoint,
        meta: &CheckpointMetadata,
        writes: &[PendingWrite],
    ) -> GraphResult<RunnableConfig>;

    async fn get_tuple(&self, cfg: &RunnableConfig) -> GraphResult<Option<CheckpointTuple>>;

    fn list<'a>(
        &'a self,
        cfg: &'a RunnableConfig,
        filter: ListFilter,
    ) -> BoxStream<'a, GraphResult<CheckpointTuple>>;

    async fn put_writes(
        &self,
        cfg: &RunnableConfig,
        writes: &[PendingWrite],
        task_id: &str,
    ) -> GraphResult<()>;

    /// Optional schema setup hook. No-op for memory; runs DDL for SQL backends.
    async fn setup(&self) -> GraphResult<()> {
        Ok(())
    }
}
