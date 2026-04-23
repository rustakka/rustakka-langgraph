//! # rustakka-langgraph-core
//!
//! Native-Rust port of LangGraph's Pregel/Bulk-Synchronous-Parallel execution
//! engine, built on top of the [`rustakka_core`] actor runtime.
//!
//! ## Subsystems
//!
//! - [`channel`] — typed `Channel` trait + reducers (LastValue, Topic,
//!   BinaryOperatorAggregate, EphemeralValue, AnyValue).
//! - [`state`]   — `GraphState` schema trait and runtime state container.
//! - [`graph`]   — `StateGraph` builder + `CompiledStateGraph` handle.
//! - [`coordinator`] — the `GraphCoordinator` actor implementing Plan/Execute/
//!   Update supersteps.
//! - [`node`]    — `NodeKind` enum (`Rust` / `Python` / `Subgraph`).
//! - [`runner`]  — public `invoke` / `stream` / `batch` entry points.
//! - [`stream`]  — `StreamEvent` and the `StreamBus` actor.
//! - [`command`] — `Command`, `Send`, `Interrupt` control-flow primitives.
//! - [`config`]  — `RunnableConfig` (mirrors upstream).
//! - [`errors`]  — graph errors mapped to upstream `langgraph.errors`.
//!
//! See `docs/index.md` for the architectural overview.

#![forbid(unsafe_code)]

pub mod channel;
pub mod command;
pub mod config;
pub mod context;
pub mod coordinator;
pub mod errors;
pub mod graph;
pub mod node;
pub mod node_worker;
pub mod runner;
pub mod state;
pub mod stream;
pub mod visualize;

pub mod prelude {
    //! Common imports for graph authors.
    pub use crate::channel::{Channel, ChannelKind, ChannelValue};
    pub use crate::command::{Command, Interrupt, Send};
    pub use crate::config::{RunnableConfig, StreamMode};
    pub use crate::errors::{GraphError, GraphResult};
    pub use crate::graph::{
        CachePolicy, CompileConfig, CompiledStateGraph, Durability, RetryPolicy, StateGraph, END,
        START,
    };
    pub use crate::node::{NodeFn, NodeKind, NodeOutput};
    pub use crate::state::{ChannelSpec, GraphState, GraphValues};
    pub use crate::stream::StreamEvent;
    pub use async_trait::async_trait;
}

#[cfg(feature = "macros")]
pub use rustakka_langgraph_macros::GraphState;
