//! `BaseStore` trait + `InMemoryStore`.
//! Mirrors `langgraph.store.base.BaseStore` and `langgraph.store.memory.InMemoryStore`.

pub mod accessor;
pub mod base;
pub mod embedding;
pub mod memory;

pub use accessor::store_accessor;
pub use base::{BaseStore, Item, ListNamespacesFilter, PutOptions, SearchHit};
pub use embedding::{Embedder, HashingEmbedder};
pub use memory::InMemoryStore;
