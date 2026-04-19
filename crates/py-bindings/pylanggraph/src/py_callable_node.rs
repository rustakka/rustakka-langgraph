//! `PyCallableNode` — wraps a Python `callable` so it can be a graph node.
//!
//! Implements `rustakka_langgraph_core::node::PyCallableNode`. We acquire the
//! GIL only to invoke the user callable; everything else (engine, channels,
//! routing) stays GIL-free per the spec.
//!
//! Async callables are detected via `inspect.iscoroutine` and bridged using
//! `pyo3_async_runtimes::tokio::into_future`.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use rustakka_langgraph_core::errors::{GraphError, GraphResult};
use rustakka_langgraph_core::node::{NodeKind, NodeOutput, PyCallableNode};

use crate::conversions::{json_to_py, py_to_json};

pub struct PyCallable {
    pub callable: PyObject,
}

impl PyCallable {
    pub fn into_node(self) -> NodeKind {
        NodeKind::Python(Arc::new(self) as Arc<dyn PyCallableNode>)
    }

    fn call_sync(
        &self,
        py: Python<'_>,
        input: BTreeMap<String, Value>,
    ) -> PyResult<NodeOutput> {
        let d = PyDict::new_bound(py);
        for (k, v) in &input {
            d.set_item(k, json_to_py(py, v)?)?;
        }
        let res = self.callable.bind(py).call1((d,))?;
        let inspect = py.import_bound("inspect")?;
        let is_coro: bool = inspect.call_method1("iscoroutine", (&res,))?.extract()?;
        if is_coro {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "async node returned a coroutine on the sync path",
            ));
        }
        materialize(py, res)
    }
}

impl PyCallableNode for PyCallable {
    fn call(
        &self,
        input: BTreeMap<String, Value>,
    ) -> Pin<Box<dyn std::future::Future<Output = GraphResult<NodeOutput>> + Send + 'static>> {
        // Acquire GIL just long enough to invoke the callable. If it returns a
        // coroutine, we transcribe it into a Rust future and release the GIL.
        let cb = Python::with_gil(|py| self.callable.clone_ref(py));
        Box::pin(async move {
            // Step 1 (GIL): call the function.
            let pending = Python::with_gil(|py| -> PyResult<Either> {
                let d = PyDict::new_bound(py);
                for (k, v) in &input {
                    d.set_item(k, json_to_py(py, v)?)?;
                }
                let res = cb.bind(py).call1((d,))?;
                let inspect = py.import_bound("inspect")?;
                let is_coro: bool = inspect.call_method1("iscoroutine", (&res,))?.extract()?;
                if is_coro {
                    let fut = pyo3_async_runtimes::tokio::into_future(res)?;
                    Ok(Either::Future(Box::pin(fut)))
                } else {
                    Ok(Either::Result(materialize(py, res)?))
                }
            })
            .map_err(|e| GraphError::Node {
                node: "<py>".into(),
                source: anyhow::anyhow!(e.to_string()),
            })?;
            match pending {
                Either::Result(out) => Ok(out),
                Either::Future(fut) => {
                    let py_obj = fut.await.map_err(|e| GraphError::Node {
                        node: "<py>".into(),
                        source: anyhow::anyhow!(e.to_string()),
                    })?;
                    Python::with_gil(|py| materialize(py, py_obj.into_bound(py)))
                        .map_err(|e| GraphError::Node {
                            node: "<py>".into(),
                            source: anyhow::anyhow!(e.to_string()),
                        })
                }
            }
        })
    }
}

enum Either {
    Result(NodeOutput),
    Future(Pin<Box<dyn std::future::Future<Output = PyResult<PyObject>> + Send>>),
}

fn materialize(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<NodeOutput> {
    if obj.is_none() {
        return Ok(NodeOutput::Halt);
    }
    // Detect Command class via attribute presence.
    if obj.hasattr("__rustakka_langgraph_command__")? {
        let cmd_json: String = obj.getattr("__rustakka_langgraph_command__")?.extract()?;
        let cmd: rustakka_langgraph_core::command::Command =
            serde_json::from_str(&cmd_json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        return Ok(NodeOutput::Command(cmd));
    }
    // Plain dict-like update.
    let v = py_to_json(&obj)?;
    Ok(NodeOutput::from_value(v))
}

// ensure the type is referenced even when unused
const _: fn() = || {
    let _ = std::any::TypeId::of::<PyCallable>();
};

// silence "unused import" for `PyCallable::call_sync` until exposed
impl PyCallable {
    #[allow(dead_code)]
    fn _keep_sync(&self, py: Python<'_>) -> PyResult<NodeOutput> {
        self.call_sync(py, BTreeMap::new())
    }
}
