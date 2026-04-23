//! Prebuilt agent factories. Mirrors `langgraph.prebuilt.*`.
//!
//! - [`tool_node::ToolNode`]      — wraps a registry of tools as a node.
//! - [`tools_condition`]          — router predicate selecting "tools" vs END.
//! - [`react_agent::create_react_agent`] — assembles a ReAct-style loop
//!   `agent → (tools? → agent : end)` from a bare LLM callback + tools.

pub mod react_agent;
pub mod supervisor;
pub mod swarm;
pub mod tool_node;

#[cfg(feature = "providers")]
pub mod providers_adapter;

pub use react_agent::{create_react_agent, ReactAgentOptions};
pub use supervisor::{create_supervisor, Agent, SupervisorRouter};
pub use swarm::create_swarm;
pub use tool_node::{tools_condition, Tool, ToolCall, ToolNode};

#[cfg(feature = "providers")]
pub use providers_adapter::{chat_model_fn, InvocationMode};
