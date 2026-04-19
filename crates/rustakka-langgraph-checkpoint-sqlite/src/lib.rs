//! SQLite-backed checkpointer. Mirrors `langgraph_checkpoint_sqlite.SqliteSaver`.
//!
//! Schema follows upstream's table layout so existing DBs interchange:
//!   `checkpoints(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
//!                parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB,
//!                metadata BLOB, PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id))`
//!   `checkpoint_writes(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
//!                      task_id TEXT, idx INTEGER, channel TEXT, type TEXT,
//!                      value BLOB, PRIMARY KEY(thread_id, checkpoint_ns,
//!                      checkpoint_id, task_id, idx))`
//!   `checkpoint_migrations(v INTEGER PRIMARY KEY)`
//!
//! Values are serialized as JSON for portability with the Python shim.

use std::str::FromStr;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;

use rustakka_langgraph_checkpoint::base::{
    Checkpoint, CheckpointMetadata, CheckpointTuple, Checkpointer, ListFilter, PendingWrite,
};
use rustakka_langgraph_core::config::RunnableConfig;
use rustakka_langgraph_core::errors::{GraphError, GraphResult};

const MIGRATIONS: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY)",
    "CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        type TEXT,
        checkpoint BLOB NOT NULL,
        metadata BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )",
    "CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        channel TEXT NOT NULL,
        type TEXT,
        value BLOB NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )",
    "CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        version TEXT NOT NULL,
        type TEXT,
        blob BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )",
];

pub struct SqliteSaver {
    pool: SqlitePool,
}

impl SqliteSaver {
    /// Open or create a database at `url` (e.g. `sqlite://./graph.db` or
    /// `sqlite::memory:`).
    pub async fn from_url(url: &str) -> GraphResult<Self> {
        let opts = SqliteConnectOptions::from_str(url)
            .map_err(|e| GraphError::other(e.to_string()))?
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let me = Self { pool };
        me.setup().await?;
        Ok(me)
    }

    pub async fn in_memory() -> GraphResult<Self> {
        Self::from_url("sqlite::memory:").await
    }
}

#[async_trait]
impl Checkpointer for SqliteSaver {
    async fn setup(&self) -> GraphResult<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        for stmt in MIGRATIONS {
            sqlx::query(stmt)
                .execute(&mut *tx)
                .await
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        }
        sqlx::query("INSERT OR IGNORE INTO checkpoint_migrations(v) VALUES (1)")
            .execute(&mut *tx)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
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
        let ckpt_blob = serde_json::to_vec(ckpt).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let meta_blob = serde_json::to_vec(meta).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        sqlx::query(
            "INSERT OR REPLACE INTO checkpoints
              (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
               type, checkpoint, metadata, created_at)
             VALUES (?, ?, ?, ?, 'json', ?, ?, ?)",
        )
        .bind(&ckpt.thread_id)
        .bind(&ckpt.checkpoint_ns)
        .bind(&ckpt.id)
        .bind(&ckpt.parent_checkpoint_id)
        .bind(&ckpt_blob)
        .bind(&meta_blob)
        .bind(ckpt.created_at.to_rfc3339())
        .execute(&mut *tx)
        .await
        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        for (i, w) in writes.iter().enumerate() {
            let v = serde_json::to_vec(&w.value).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            sqlx::query(
                "INSERT OR REPLACE INTO checkpoint_writes
                  (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
                 VALUES (?, ?, ?, ?, ?, ?, 'json', ?)",
            )
            .bind(&ckpt.thread_id)
            .bind(&ckpt.checkpoint_ns)
            .bind(&ckpt.id)
            .bind(&w.task_id)
            .bind(i as i64)
            .bind(&w.channel)
            .bind(&v)
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
        let row = if let Some(id) = &cfg.checkpoint_id {
            sqlx::query(
                "SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata
                 FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
            )
            .bind(tid)
            .bind(ns)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
        } else {
            sqlx::query(
                "SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata
                 FROM checkpoints WHERE thread_id=? AND checkpoint_ns=?
                 ORDER BY created_at DESC LIMIT 1",
            )
            .bind(tid)
            .bind(ns)
            .fetch_optional(&self.pool)
            .await
        }
        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let Some(row) = row else { return Ok(None) };
        let ckpt_blob: Vec<u8> = row.get("checkpoint");
        let meta_blob: Vec<u8> = row.get("metadata");
        let checkpoint: Checkpoint =
            serde_json::from_slice(&ckpt_blob).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let metadata: CheckpointMetadata =
            serde_json::from_slice(&meta_blob).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let mut new_cfg = cfg.clone();
        new_cfg.checkpoint_id = Some(checkpoint.id.clone());
        // pending writes
        let writes = sqlx::query(
            "SELECT task_id, channel, value FROM checkpoint_writes
             WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? ORDER BY idx",
        )
        .bind(tid)
        .bind(ns)
        .bind(&checkpoint.id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let pending_writes: Vec<PendingWrite> = writes
            .into_iter()
            .map(|r| {
                let v: Vec<u8> = r.get("value");
                PendingWrite {
                    task_id: r.get("task_id"),
                    channel: r.get("channel"),
                    value: serde_json::from_slice(&v).unwrap_or(serde_json::Value::Null),
                }
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
        let fut = async move {
            let tid = cfg.thread_id().unwrap_or_default().to_string();
            let ns = cfg.checkpoint_ns().to_string();
            let limit = filter.limit.unwrap_or(u32::MAX) as i64;
            let rows = sqlx::query(
                "SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata, created_at
                 FROM checkpoints WHERE thread_id=? AND checkpoint_ns=?
                 ORDER BY created_at DESC LIMIT ?",
            )
            .bind(&tid)
            .bind(&ns)
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
                let ckpt_blob: Vec<u8> = row.get("checkpoint");
                let meta_blob: Vec<u8> = row.get("metadata");
                let checkpoint: Checkpoint =
                    serde_json::from_slice(&ckpt_blob)
                        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
                let metadata: CheckpointMetadata =
                    serde_json::from_slice(&meta_blob)
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
        Box::pin(futures::stream::once(async move {
            match fut.await {
                Ok(v) => futures::stream::iter(v),
                Err(e) => futures::stream::iter(vec![Err(e)]),
            }
        })
        .flatten())
    }

    async fn put_writes(
        &self,
        cfg: &RunnableConfig,
        writes: &[PendingWrite],
        task_id: &str,
    ) -> GraphResult<()> {
        let tid = cfg.thread_id().unwrap_or_default().to_string();
        let ns = cfg.checkpoint_ns().to_string();
        let id = cfg.checkpoint_id.clone().unwrap_or_default();
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        for (i, w) in writes.iter().enumerate() {
            let v = serde_json::to_vec(&w.value).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            sqlx::query(
                "INSERT OR REPLACE INTO checkpoint_writes
                   (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
                 VALUES (?, ?, ?, ?, ?, ?, 'json', ?)",
            )
            .bind(&tid)
            .bind(&ns)
            .bind(&id)
            .bind(task_id)
            .bind(i as i64)
            .bind(&w.channel)
            .bind(&v)
            .execute(&mut *tx)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        }
        tx.commit().await.map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustakka_langgraph_checkpoint::memory::make_checkpoint;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn put_and_get_roundtrip() {
        let saver = SqliteSaver::in_memory().await.unwrap();
        let cfg = RunnableConfig::with_thread("t1");
        let mut values = BTreeMap::new();
        values.insert("x".into(), json!(1));
        let ckpt = make_checkpoint(&cfg, 1, &values, &BTreeMap::new(), None);
        let meta = CheckpointMetadata { source: "input".into(), step: 1, ..Default::default() };
        let new_cfg = saver.put(&cfg, &ckpt, &meta, &[]).await.unwrap();
        let tup = saver.get_tuple(&new_cfg).await.unwrap().unwrap();
        assert_eq!(tup.checkpoint.channel_values["x"], json!(1));
    }
}
