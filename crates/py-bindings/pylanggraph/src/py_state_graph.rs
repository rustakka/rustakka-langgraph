//! `PyStateGraph` — Python-facing builder mirroring `langgraph.graph.StateGraph`.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde_json::Value;

use rustakka_langgraph_core::graph::{
    CompileConfig, CompiledStateGraph, RouterFn, StateGraph, END as RA_END, START as RA_START,
};
use rustakka_langgraph_core::node::NodeKind;
use rustakka_langgraph_core::state::{ChannelSpec, DynamicState};

use crate::conversions::map_to_pydict;
use crate::py_callable_node::PyCallable;
use crate::py_compiled_state_graph::PyCompiledStateGraph;

#[pyclass(name = "StateGraph", module = "rustakka_langgraph._native")]
pub struct PyStateGraph {
    inner: Option<StateGraph<DynamicState>>,
}

#[pymethods]
impl PyStateGraph {
    #[new]
    #[pyo3(signature = (state_schema=None, _config_schema=None, _input=None, _output=None))]
    fn new(
        state_schema: Option<Bound<'_, PyAny>>,
        _config_schema: Option<Bound<'_, PyAny>>,
        _input: Option<Bound<'_, PyAny>>,
        _output: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut g = StateGraph::<DynamicState>::new();
        // If the schema is a dict-like with `__annotations__` containing
        // `Annotated[..., "add_messages"]` etc., extract reducer hints.
        if let Some(schema) = state_schema {
            if let Ok(annotations) = schema.getattr("__annotations__") {
                if let Ok(d) = annotations.downcast::<PyDict>() {
                    for (k, v) in d.iter() {
                        let name: String = k.extract()?;
                        let reducer = extract_reducer(&v).unwrap_or_else(|| "last_value".into());
                        g.add_channel(ChannelSpec { name, reducer });
                    }
                }
            }
        }
        Ok(Self { inner: Some(g) })
    }

    #[pyo3(signature = (name, action, _metadata=None))]
    fn add_node(
        &mut self,
        name: String,
        action: Bound<'_, PyAny>,
        _metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let g = self.inner.as_mut().ok_or_else(already_compiled)?;
        let node = PyCallable { callable: action.unbind() }.into_node();
        g.add_node(name, node).map_err(graph_err)?;
        Ok(())
    }

    fn add_edge(&mut self, source: String, target: String) -> PyResult<()> {
        let g = self.inner.as_mut().ok_or_else(already_compiled)?;
        g.add_edge(source, target);
        Ok(())
    }

    #[pyo3(signature = (source, path, path_map=None, _then=None))]
    fn add_conditional_edges(
        &mut self,
        source: String,
        path: Bound<'_, PyAny>,
        path_map: Option<Bound<'_, PyDict>>,
        _then: Option<String>,
    ) -> PyResult<()> {
        let g = self.inner.as_mut().ok_or_else(already_compiled)?;
        let path_obj: PyObject = path.into();
        let router: RouterFn = Arc::new(move |values| {
            Python::with_gil(|py| -> PyResult<Vec<String>> {
                let d = map_to_pydict(py, values)?;
                let res = path_obj.bind(py).call1(PyTuple::new_bound(py, [d.into_any()]))?;
                if let Ok(s) = res.extract::<String>() {
                    return Ok(vec![s]);
                }
                if let Ok(v) = res.extract::<Vec<String>>() {
                    return Ok(v);
                }
                Ok(vec![res.str()?.to_string()])
            })
            .unwrap_or_default()
        });
        let branches = if let Some(m) = path_map {
            let mut out = HashMap::new();
            for (k, v) in m.iter() {
                out.insert(k.extract()?, v.extract()?);
            }
            Some(out)
        } else {
            None
        };
        g.add_conditional_edges(source, router, branches);
        Ok(())
    }

    fn set_entry_point(&mut self, name: String) -> PyResult<()> {
        let g = self.inner.as_mut().ok_or_else(already_compiled)?;
        g.set_entry_point(name);
        Ok(())
    }

    fn set_finish_point(&mut self, name: String) -> PyResult<()> {
        let g = self.inner.as_mut().ok_or_else(already_compiled)?;
        g.set_finish_point(name);
        Ok(())
    }

    #[pyo3(signature = (checkpointer=None, store=None, debug=None, _interrupt_before=None, _interrupt_after=None))]
    fn compile(
        &mut self,
        py: Python<'_>,
        checkpointer: Option<Bound<'_, PyAny>>,
        store: Option<Bound<'_, PyAny>>,
        debug: Option<bool>,
        _interrupt_before: Option<Bound<'_, PyAny>>,
        _interrupt_after: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyCompiledStateGraph> {
        let g = self.inner.take().ok_or_else(already_compiled)?;
        let mut cfg = CompileConfig::default();
        cfg.debug = debug.unwrap_or(false);
        let compiled: CompiledStateGraph = py
            .allow_threads(|| {
                pyo3_async_runtimes::tokio::get_runtime().block_on(async { g.compile(cfg).await })
            })
            .map_err(graph_err)?;
        let mut wrapper = PyCompiledStateGraph::from_compiled(compiled);
        wrapper.attach_checkpointer(py, checkpointer)?;
        wrapper.attach_store(py, store)?;
        Ok(wrapper)
    }
}

fn already_compiled() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("StateGraph already compiled")
}

pub(crate) fn graph_err(e: rustakka_langgraph_core::errors::GraphError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

fn extract_reducer(v: &Bound<'_, PyAny>) -> Option<String> {
    // Look for typing.Annotated[..., reducer]
    let metadata = v.getattr("__metadata__").ok()?;
    let tup = metadata.downcast::<PyTuple>().ok()?;
    for item in tup.iter() {
        // Reducer can be a callable named `add_messages`, or a string.
        if let Ok(s) = item.extract::<String>() {
            return Some(s);
        }
        if let Ok(name) = item.getattr("__name__").and_then(|n| n.extract::<String>()) {
            return Some(name);
        }
    }
    None
}

// suppress unused warnings
const _: fn() = || {
    let _ = RA_START;
    let _ = RA_END;
    let _: Option<Value> = None;
};
