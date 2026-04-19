//! DDL parity with `langgraph-checkpoint-postgres`.
//!
//! Tables (in `${schema}` — defaults to `public`):
//!   - `checkpoints`
//!   - `checkpoint_writes`
//!   - `checkpoint_blobs`
//!   - `checkpoint_migrations`
//!
//! Schema name is interpolated into the statements at setup-time so we can
//! support shared databases per the LangGraph runtime guide.

pub fn migrations(schema: &str) -> Vec<String> {
    let s = schema;
    vec![
        format!("CREATE SCHEMA IF NOT EXISTS \"{s}\""),
        format!(
            "CREATE TABLE IF NOT EXISTS \"{s}\".checkpoint_migrations (
              v INTEGER PRIMARY KEY
            )"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS \"{s}\".checkpoints (
              thread_id TEXT NOT NULL,
              checkpoint_ns TEXT NOT NULL DEFAULT '',
              checkpoint_id TEXT NOT NULL,
              parent_checkpoint_id TEXT,
              type TEXT,
              checkpoint JSONB NOT NULL,
              metadata JSONB NOT NULL DEFAULT '{{}}',
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )"
        ),
        format!(
            "CREATE INDEX IF NOT EXISTS checkpoints_thread_ns_created_idx
              ON \"{s}\".checkpoints (thread_id, checkpoint_ns, created_at DESC)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS \"{s}\".checkpoint_writes (
              thread_id TEXT NOT NULL,
              checkpoint_ns TEXT NOT NULL DEFAULT '',
              checkpoint_id TEXT NOT NULL,
              task_id TEXT NOT NULL,
              idx INTEGER NOT NULL,
              channel TEXT NOT NULL,
              type TEXT,
              value JSONB NOT NULL,
              PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS \"{s}\".checkpoint_blobs (
              thread_id TEXT NOT NULL,
              checkpoint_ns TEXT NOT NULL DEFAULT '',
              channel TEXT NOT NULL,
              version TEXT NOT NULL,
              type TEXT,
              blob BYTEA,
              PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            )"
        ),
        format!(
            "INSERT INTO \"{s}\".checkpoint_migrations(v) VALUES (1) ON CONFLICT DO NOTHING"
        ),
    ]
}
