//! Runtime configuration mirroring upstream `langgraph` `RunnableConfig`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Streaming modes mirroring `langgraph.types.StreamMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamMode {
    Values,
    Updates,
    Messages,
    Custom,
    Debug,
}

impl Default for StreamMode {
    fn default() -> Self {
        StreamMode::Values
    }
}

/// LangGraph `RunnableConfig`: holds `configurable.thread_id`, recursion
/// limit, run-id, tags, metadata, and streaming preferences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunnableConfig {
    #[serde(default)]
    pub configurable: BTreeMap<String, Value>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default)]
    pub recursion_limit: Option<u32>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub stream_mode: Option<Vec<StreamMode>>,
    /// Subgraph namespace (`checkpoint_ns`).
    #[serde(default)]
    pub checkpoint_ns: Option<String>,
    /// Specific checkpoint id to resume from (time-travel).
    #[serde(default)]
    pub checkpoint_id: Option<String>,
}

impl RunnableConfig {
    pub fn with_thread(thread_id: impl Into<String>) -> Self {
        let mut me = Self::default();
        me.configurable.insert("thread_id".into(), Value::String(thread_id.into()));
        me
    }

    pub fn thread_id(&self) -> Option<&str> {
        self.configurable.get("thread_id").and_then(|v| v.as_str())
    }

    pub fn checkpoint_ns(&self) -> &str {
        self.checkpoint_ns.as_deref().unwrap_or("")
    }

    pub fn effective_recursion_limit(&self) -> u32 {
        self.recursion_limit.unwrap_or(25)
    }
}
