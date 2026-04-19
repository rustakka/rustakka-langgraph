//! Python wrappers for `Command`, `Send`, `Interrupt`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use rustakka_langgraph_core::command::{Command, Interrupt, Send as RaSend};

use crate::conversions::{json_to_py, py_to_json};

#[pyclass(name = "Command", module = "rustakka_langgraph._native")]
#[derive(Clone)]
pub struct PyCommand {
    pub inner: Command,
}

#[pymethods]
impl PyCommand {
    #[new]
    #[pyo3(signature = (update=None, goto=None, send=None, resume=None, graph=None))]
    fn new(
        update: Option<Bound<'_, PyDict>>,
        goto: Option<Bound<'_, PyAny>>,
        send: Option<Vec<PySend>>,
        resume: Option<Bound<'_, PyAny>>,
        graph: Option<String>,
    ) -> PyResult<Self> {
        let mut cmd = Command::default();
        if let Some(d) = update {
            for (k, v) in d.iter() {
                cmd.update.insert(k.extract()?, py_to_json(&v)?);
            }
        }
        if let Some(g) = goto {
            if let Ok(s) = g.extract::<String>() {
                cmd.goto.push(s);
            } else if let Ok(v) = g.extract::<Vec<String>>() {
                cmd.goto = v;
            }
        }
        if let Some(ss) = send {
            cmd.send = ss.into_iter().map(|p| p.inner).collect();
        }
        if let Some(r) = resume {
            cmd.resume = Some(py_to_json(&r)?);
        }
        cmd.graph = graph;
        Ok(Self { inner: cmd })
    }

    /// Marker attribute for `materialize` in py_callable_node.rs.
    #[getter]
    fn __rustakka_langgraph_command__(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Command({:?})", self.inner)
    }
}

#[pyclass(name = "Send", module = "rustakka_langgraph._native")]
#[derive(Clone)]
pub struct PySend {
    pub inner: RaSend,
}

#[pymethods]
impl PySend {
    #[new]
    fn new(node: String, arg: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self { inner: RaSend { node, arg: py_to_json(&arg)? } })
    }
    fn __repr__(&self) -> String {
        format!("Send(node={:?}, arg={:?})", self.inner.node, self.inner.arg)
    }
}

#[pyclass(name = "Interrupt", module = "rustakka_langgraph._native")]
#[derive(Clone)]
pub struct PyInterrupt {
    pub inner: Interrupt,
}

#[pymethods]
impl PyInterrupt {
    #[new]
    fn new(node: String, value: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self { inner: Interrupt::new(node, py_to_json(&value)?) })
    }

    #[getter]
    fn node(&self) -> &str {
        &self.inner.node
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.value)
    }
}

// Suppress "unused import" of Value
const _: fn() = || {
    let _: Option<Value> = None;
};
