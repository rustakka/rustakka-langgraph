//! Postgres-backed `BaseStore`.
//!
//! Schema parity with upstream `langgraph.store.postgres`:
//! ```sql
//! CREATE TABLE store (
//!   prefix TEXT NOT NULL,        -- '/'-joined namespace
//!   key    TEXT NOT NULL,
//!   value  JSONB NOT NULL,
//!   created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
//!   updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
//!   expires_at TIMESTAMPTZ,
//!   PRIMARY KEY (prefix, key)
//! );
//! CREATE INDEX store_prefix_idx ON store (prefix);
//! ```

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use rustakka_langgraph_core::errors::{GraphError, GraphResult};
use rustakka_langgraph_store::base::{
    BaseStore, Item, ListNamespacesFilter, Namespace, PutOptions, SearchHit,
};

pub struct PostgresStore {
    pool: PgPool,
    schema: String,
}

pub type AsyncPostgresStore = PostgresStore;

impl PostgresStore {
    pub async fn from_url(url: &str) -> GraphResult<Self> {
        Self::from_url_with_schema(url, "public").await
    }

    pub async fn from_url_with_schema(url: &str, schema: &str) -> GraphResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        let me = Self { pool, schema: sanitize(schema) };
        me.setup().await?;
        Ok(me)
    }

    fn t(&self) -> String {
        format!("\"{}\".store", self.schema)
    }
}

fn sanitize(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect()
}

fn join(ns: &Namespace) -> String {
    ns.join("/")
}

fn split(prefix: &str) -> Namespace {
    if prefix.is_empty() {
        Vec::new()
    } else {
        prefix.split('/').map(|s| s.to_string()).collect()
    }
}

#[async_trait]
impl BaseStore for PostgresStore {
    async fn setup(&self) -> GraphResult<()> {
        let stmts = vec![
            format!("CREATE SCHEMA IF NOT EXISTS \"{}\"", self.schema),
            format!(
                "CREATE TABLE IF NOT EXISTS {} (
                  prefix TEXT NOT NULL,
                  key    TEXT NOT NULL,
                  value  JSONB NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  expires_at TIMESTAMPTZ,
                  PRIMARY KEY (prefix, key)
                )",
                self.t()
            ),
            format!("CREATE INDEX IF NOT EXISTS store_prefix_idx ON {} (prefix)", self.t()),
        ];
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        for s in stmts {
            sqlx::query(&s)
                .execute(&mut *tx)
                .await
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        }
        tx.commit().await.map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, namespace: &Namespace, key: &str) -> GraphResult<Option<Item>> {
        let q = format!(
            "SELECT prefix, key, value, created_at, updated_at, expires_at FROM {}
             WHERE prefix=$1 AND key=$2 AND (expires_at IS NULL OR expires_at > now())",
            self.t()
        );
        let row = sqlx::query(&q)
            .bind(join(namespace))
            .bind(key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(row.map(|r| Item {
            namespace: split(&r.get::<String, _>("prefix")),
            key: r.get("key"),
            value: r.get("value"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
            expires_at: r.try_get::<Option<DateTime<Utc>>, _>("expires_at").unwrap_or(None),
        }))
    }

    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: Value,
        opts: PutOptions,
    ) -> GraphResult<()> {
        let expires_at = opts.ttl_seconds.map(|s| Utc::now() + Duration::seconds(s as i64));
        let q = format!(
            "INSERT INTO {} (prefix, key, value, expires_at)
             VALUES ($1,$2,$3,$4)
             ON CONFLICT (prefix, key) DO UPDATE
               SET value=EXCLUDED.value, updated_at=now(), expires_at=EXCLUDED.expires_at",
            self.t()
        );
        sqlx::query(&q)
            .bind(join(namespace))
            .bind(key)
            .bind(&value)
            .bind(expires_at)
            .execute(&self.pool)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, namespace: &Namespace, key: &str) -> GraphResult<()> {
        let q = format!("DELETE FROM {} WHERE prefix=$1 AND key=$2", self.t());
        sqlx::query(&q)
            .bind(join(namespace))
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        namespace_prefix: &Namespace,
        query: Option<&str>,
        limit: u32,
        offset: u32,
    ) -> GraphResult<Vec<SearchHit>> {
        let prefix = join(namespace_prefix);
        let q = format!(
            "SELECT prefix, key, value, created_at, updated_at, expires_at FROM {}
             WHERE prefix LIKE $1 AND (expires_at IS NULL OR expires_at > now())
              AND ($2::text IS NULL OR value::text ILIKE '%' || $2 || '%')
             ORDER BY updated_at DESC LIMIT $3 OFFSET $4",
            self.t()
        );
        let rows = sqlx::query(&q)
            .bind(format!("{prefix}%"))
            .bind(query)
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(rows
            .into_iter()
            .map(|r| SearchHit {
                item: Item {
                    namespace: split(&r.get::<String, _>("prefix")),
                    key: r.get("key"),
                    value: r.get("value"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    expires_at: r.try_get::<Option<DateTime<Utc>>, _>("expires_at").unwrap_or(None),
                },
                score: None,
            })
            .collect())
    }

    async fn list_namespaces(&self, filter: ListNamespacesFilter) -> GraphResult<Vec<Namespace>> {
        let prefix = filter.prefix.as_ref().map(|p| join(p)).unwrap_or_default();
        let q = format!(
            "SELECT DISTINCT prefix FROM {} WHERE prefix LIKE $1 ORDER BY prefix LIMIT $2",
            self.t()
        );
        let rows = sqlx::query(&q)
            .bind(format!("{prefix}%"))
            .bind(filter.limit.unwrap_or(1000) as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(rows
            .into_iter()
            .map(|r| split(&r.get::<String, _>("prefix")))
            .filter(|ns| filter.max_depth.map(|d| (ns.len() as u32) <= d).unwrap_or(true))
            .collect())
    }
}
