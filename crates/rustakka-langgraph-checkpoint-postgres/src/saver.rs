//! `PostgresSaver` + `AsyncPostgresSaver`.
//!
//! Both share a single `sqlx::PgPool` under the hood; the distinction in
//! upstream Python (`PostgresSaver` vs `AsyncPostgresSaver`) is preserved
//! here as separate type aliases so the Python shim can re-export both.

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use rustakka_langgraph_checkpoint::base::{
    Checkpoint, CheckpointMetadata, CheckpointTuple, Checkpointer, ListFilter, PendingWrite,
};
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::errors::{GraphError, GraphResult};

use crate::schema::migrations;

pub struct PostgresSaver {
    pool: PgPool,
    schema: String,
}

pub type AsyncPostgresSaver = PostgresSaver;

impl PostgresSaver {
    pub async fn from_url(url: &str) -> GraphResult<Self> {
        Self::from_url_with_schema(url, "public").await
    }

    pub async fn from_url_with_schema(url: &str, schema: &str) -> GraphResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let me = Self { pool, schema: sanitize_schema(schema) };
        me.setup().await?;
        Ok(me)
    }

    fn t(&self, table: &str) -> String {
        format!("\"{}\".{}", self.schema, table)
    }
}

fn sanitize_schema(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect()
}

#[async_trait]
impl Checkpointer for PostgresSaver {
    async fn setup(&self) -> GraphResult<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        for stmt in migrations(&self.schema) {
            sqlx::query(&stmt)
                .execute(&mut *tx)
                .await
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        }
        tx.commit().await.map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(())
    }

    async fn put(
        &self,
        cfg: &RunnableConfig,
        ckpt: &Checkpoint,
        meta: &CheckpointMetadata,
        writes: &[PendingWrite],
    ) -> GraphResult<RunnableConfig> {
        let ckpt_json = serde_json::to_value(ckpt).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let meta_json = serde_json::to_value(meta).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let q = format!(
            "INSERT INTO {} (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at)
             VALUES ($1,$2,$3,$4,'json',$5,$6,$7)
             ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE
             SET checkpoint=EXCLUDED.checkpoint, metadata=EXCLUDED.metadata",
            self.t("checkpoints")
        );
        sqlx::query(&q)
            .bind(&ckpt.thread_id)
            .bind(&ckpt.checkpoint_ns)
            .bind(&ckpt.id)
            .bind(&ckpt.parent_checkpoint_id)
            .bind(&ckpt_json)
            .bind(&meta_json)
            .bind(ckpt.created_at)
            .execute(&mut *tx)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let qw = format!(
            "INSERT INTO {} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
             VALUES ($1,$2,$3,$4,$5,$6,'json',$7)
             ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET value=EXCLUDED.value",
            self.t("checkpoint_writes")
        );
        for (i, w) in writes.iter().enumerate() {
            sqlx::query(&qw)
                .bind(&ckpt.thread_id)
                .bind(&ckpt.checkpoint_ns)
                .bind(&ckpt.id)
                .bind(&w.task_id)
                .bind(i as i32)
                .bind(&w.channel)
                .bind(&w.value)
                .execute(&mut *tx)
                .await
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        }
        tx.commit().await.map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let mut new_cfg = cfg.clone();
        new_cfg.checkpoint_id = Some(ckpt.id.clone());
        Ok(new_cfg)
    }

    async fn get_tuple(&self, cfg: &RunnableConfig) -> GraphResult<Option<CheckpointTuple>> {
        let tid = cfg.thread_id().unwrap_or_default();
        let ns = cfg.checkpoint_ns();
        let q_latest = format!(
            "SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM {}
             WHERE thread_id=$1 AND checkpoint_ns=$2
             ORDER BY created_at DESC LIMIT 1",
            self.t("checkpoints")
        );
        let q_byid = format!(
            "SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM {}
             WHERE thread_id=$1 AND checkpoint_ns=$2 AND checkpoint_id=$3",
            self.t("checkpoints")
        );
        let row = if let Some(id) = &cfg.checkpoint_id {
            sqlx::query(&q_byid).bind(tid).bind(ns).bind(id).fetch_optional(&self.pool).await
        } else {
            sqlx::query(&q_latest).bind(tid).bind(ns).fetch_optional(&self.pool).await
        }
        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let Some(row) = row else { return Ok(None) };
        let ckpt_json: serde_json::Value = row.get("checkpoint");
        let meta_json: serde_json::Value = row.get("metadata");
        let checkpoint: Checkpoint = serde_json::from_value(ckpt_json)
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let metadata: CheckpointMetadata = serde_json::from_value(meta_json)
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let mut new_cfg = cfg.clone();
        new_cfg.checkpoint_id = Some(checkpoint.id.clone());
        let qw = format!(
            "SELECT task_id, channel, value FROM {} WHERE thread_id=$1 AND checkpoint_ns=$2 AND checkpoint_id=$3 ORDER BY idx",
            self.t("checkpoint_writes")
        );
        let writes = sqlx::query(&qw)
            .bind(tid)
            .bind(ns)
            .bind(&checkpoint.id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let pending_writes: Vec<PendingWrite> = writes
            .into_iter()
            .map(|r| PendingWrite {
                task_id: r.get("task_id"),
                channel: r.get("channel"),
                value: r.get("value"),
            })
            .collect();
        Ok(Some(CheckpointTuple {
            config: new_cfg,
            checkpoint,
            metadata,
            pending_writes,
            parent_config: None,
        }))
    }

    fn list<'a>(
        &'a self,
        cfg: &'a RunnableConfig,
        filter: ListFilter,
    ) -> BoxStream<'a, GraphResult<CheckpointTuple>> {
        let pool = self.pool.clone();
        let cfg = cfg.clone();
        let table = self.t("checkpoints");
        let fut = async move {
            let q = format!(
                "SELECT checkpoint_id, checkpoint, metadata, created_at FROM {table}
                 WHERE thread_id=$1 AND checkpoint_ns=$2
                 ORDER BY created_at DESC LIMIT $3"
            );
            let limit = filter.limit.unwrap_or(u32::MAX) as i64;
            let rows = sqlx::query(&q)
                .bind(cfg.thread_id().unwrap_or_default())
                .bind(cfg.checkpoint_ns())
                .bind(limit)
                .fetch_all(&pool)
                .await
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            let mut out: Vec<GraphResult<CheckpointTuple>> = Vec::new();
            let mut skipping = filter.before.is_some();
            for row in rows {
                let id: String = row.get("checkpoint_id");
                if skipping {
                    if Some(&id) == filter.before.as_ref() {
                        skipping = false;
                    }
                    continue;
                }
                let ckpt_json: serde_json::Value = row.get("checkpoint");
                let meta_json: serde_json::Value = row.get("metadata");
                let checkpoint: Checkpoint = serde_json::from_value(ckpt_json)
                    .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
                let metadata: CheckpointMetadata = serde_json::from_value(meta_json)
                    .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
                let mut new_cfg = cfg.clone();
                new_cfg.checkpoint_id = Some(checkpoint.id.clone());
                out.push(Ok(CheckpointTuple {
                    config: new_cfg,
                    checkpoint,
                    metadata,
                    pending_writes: Vec::new(),
                    parent_config: None,
                }));
            }
            Ok::<Vec<GraphResult<CheckpointTuple>>, GraphError>(out)
        };
        Box::pin(
            futures::stream::once(async move {
                match fut.await {
                    Ok(v) => futures::stream::iter(v),
                    Err(e) => futures::stream::iter(vec![Err(e)]),
                }
            })
            .flatten(),
        )
    }

    async fn put_writes(
        &self,
        cfg: &RunnableConfig,
        writes: &[PendingWrite],
        task_id: &str,
    ) -> GraphResult<()> {
        let q = format!(
            "INSERT INTO {} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
             VALUES ($1,$2,$3,$4,$5,$6,'json',$7)
             ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET value=EXCLUDED.value",
            self.t("checkpoint_writes")
        );
        let tid = cfg.thread_id().unwrap_or_default();
        let ns = cfg.checkpoint_ns();
        let id = cfg.checkpoint_id.clone().unwrap_or_default();
        for (i, w) in writes.iter().enumerate() {
            sqlx::query(&q)
                .bind(tid)
                .bind(ns)
                .bind(&id)
                .bind(task_id)
                .bind(i as i32)
                .bind(&w.channel)
                .bind(&w.value)
                .execute(&self.pool)
                .await
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        }
        Ok(())
    }
}
