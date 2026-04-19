//! Adapter implementing `CheckpointerHook` (consumed by the coordinator) on
//! top of any `Checkpointer`. This keeps the core engine free of checkpointer
//! implementation details.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rustakka_langgraph_core::channel::ChannelSnapshot;
use rustakka_langgraph_core::command::Interrupt;
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::coordinator::{CheckpointReplay, CheckpointerHook};
use rustakka_langgraph_core::errors::GraphResult;

use crate::base::{CheckpointMetadata, Checkpointer, PendingWrite};
use crate::memory::make_checkpoint;

/// Wrap a `Checkpointer` for use by the engine.
pub struct CheckpointerHookAdapter<C: Checkpointer> {
    inner: Arc<C>,
}

impl<C: Checkpointer> CheckpointerHookAdapter<C> {
    pub fn new(inner: Arc<C>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[async_trait]
impl<C: Checkpointer> CheckpointerHook for CheckpointerHookAdapter<C> {
    async fn put_step(
        &self,
        cfg: &RunnableConfig,
        step: u64,
        values: &BTreeMap<String, Value>,
        snapshot: &BTreeMap<String, ChannelSnapshot>,
        pending_writes: &[(String, BTreeMap<String, Value>)],
        interrupt: Option<&Interrupt>,
    ) -> GraphResult<()> {
        let ckpt = make_checkpoint(cfg, step, values, snapshot, interrupt);
        let mut writes_map: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
        let mut pw: Vec<PendingWrite> = Vec::new();
        for (node, w) in pending_writes {
            writes_map.insert(node.clone(), w.clone());
            for (ch, v) in w {
                pw.push(PendingWrite {
                    task_id: node.clone(),
                    channel: ch.clone(),
                    value: v.clone(),
                });
            }
        }
        let meta = CheckpointMetadata {
            source: if step == 1 { "input".into() } else { "loop".into() },
            step,
            writes: writes_map,
            parents: Default::default(),
        };
        self.inner.put(cfg, &ckpt, &meta, &pw).await?;
        Ok(())
    }

    async fn get_latest(&self, cfg: &RunnableConfig) -> GraphResult<Option<CheckpointReplay>> {
        let Some(tup) = self.inner.get_tuple(cfg).await? else { return Ok(None) };
        Ok(Some(CheckpointReplay {
            step: tup.checkpoint.step,
            snapshot: tup.checkpoint.channel_snapshots,
            interrupt: tup.checkpoint.interrupt,
        }))
    }
}
