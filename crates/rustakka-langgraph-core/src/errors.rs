//! Graph errors. Mirrors `langgraph.errors`:
//! `GraphRecursionError`, `InvalidUpdateError`, `NodeInterrupt`,
//! `EmptyInputError`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("graph reached recursion limit of {limit} supersteps without halting")]
    Recursion { limit: u32 },

    #[error("invalid update for channel `{channel}`: {reason}")]
    InvalidUpdate { channel: String, reason: String },

    #[error("graph received empty input")]
    EmptyInput,

    #[error("node `{node}` was interrupted")]
    NodeInterrupt { node: String, payload: serde_json::Value },

    #[error("unknown node `{0}` referenced in edges or compile config")]
    UnknownNode(String),

    #[error("duplicate node `{0}`")]
    DuplicateNode(String),

    #[error("graph has no entry point — call set_entry_point or add_edge(START, ...)")]
    MissingEntryPoint,

    #[error("compile error: {0}")]
    Compile(String),

    #[error("runtime error in node `{node}`: {source}")]
    Node {
        node: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("checkpointer error: {0}")]
    Checkpoint(String),

    #[error("internal coordinator error: {0}")]
    Coordinator(String),

    #[error("{0}")]
    Other(String),
}

pub type GraphResult<T> = Result<T, GraphError>;

impl GraphError {
    pub fn other(msg: impl Into<String>) -> Self {
        GraphError::Other(msg.into())
    }
}
