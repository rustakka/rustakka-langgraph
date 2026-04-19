//! `BaseStore` trait + `InMemoryStore`.
//! Mirrors `langgraph.store.base.BaseStore` and `langgraph.store.memory.InMemoryStore`.

pub mod base;
pub mod memory;

pub use base::{BaseStore, Item, ListNamespacesFilter, PutOptions, SearchHit};
pub use memory::InMemoryStore;
