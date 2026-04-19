//! # rustakka-langgraph
//!
//! Umbrella re-export facade. Most callers import from [`prelude`].

pub use rustakka_langgraph_core as core;
pub use rustakka_langgraph_checkpoint as checkpoint;
pub use rustakka_langgraph_store as store;

#[cfg(feature = "sqlite")]
pub use rustakka_langgraph_checkpoint_sqlite as checkpoint_sqlite;

#[cfg(feature = "postgres")]
pub use rustakka_langgraph_checkpoint_postgres as checkpoint_postgres;

#[cfg(feature = "postgres")]
pub use rustakka_langgraph_store_postgres as store_postgres;

#[cfg(feature = "prebuilt")]
pub use rustakka_langgraph_prebuilt as prebuilt;

pub mod prelude {
    pub use crate::core::prelude::*;
    pub use crate::core::runner::{invoke_dynamic, invoke_with_checkpointer, resume, stream};
    pub use crate::checkpoint::{CheckpointerHookAdapter, MemorySaver};
    pub use crate::store::{BaseStore, InMemoryStore};
}
