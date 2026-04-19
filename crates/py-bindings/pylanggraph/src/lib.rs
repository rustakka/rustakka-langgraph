//! PyO3 bindings exposing the rustakka-langgraph engine as a Python module.
//!
//! Module path: `rustakka_langgraph._native`. The pure-Python facade in
//! `python/langgraph/` re-exports these symbols under the upstream-compatible
//! `langgraph.*` API.

#![allow(clippy::module_inception)]

use pyo3::prelude::*;

mod conversions;
mod py_callable_node;
mod py_command;
mod py_compiled_state_graph;
mod py_state_graph;
mod py_savers;
mod py_stores;
mod runtime;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    runtime::init(py);
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("START", rustakka_langgraph_core::graph::START)?;
    m.add("END", rustakka_langgraph_core::graph::END)?;

    m.add_class::<py_state_graph::PyStateGraph>()?;
    m.add_class::<py_compiled_state_graph::PyCompiledStateGraph>()?;
    m.add_class::<py_command::PyCommand>()?;
    m.add_class::<py_command::PySend>()?;
    m.add_class::<py_command::PyInterrupt>()?;
    m.add_class::<py_savers::PyMemorySaver>()?;
    m.add_class::<py_stores::PyInMemoryStore>()?;

    #[cfg(feature = "sqlite")]
    m.add_class::<py_savers::PySqliteSaver>()?;

    #[cfg(feature = "postgres")]
    {
        m.add_class::<py_savers::PyPostgresSaver>()?;
        m.add_class::<py_stores::PyPostgresStore>()?;
    }

    Ok(())
}
