//! `PyCompiledStateGraph` — runtime handle exposing `invoke`/`ainvoke`/
//! `stream`/`astream`/`get_state`/`update_state`/`with_config`.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use rustakka_langgraph_checkpoint::base::Checkpointer;
use rustakka_langgraph_checkpoint::CheckpointerHookAdapter;
use rustakka_langgraph_core::config::{RunnableConfig, StreamMode};
use rustakka_langgraph_core::coordinator::CheckpointerHook;
use rustakka_langgraph_core::graph::CompiledStateGraph;
use rustakka_langgraph_core::runner::{
    invoke_dynamic, invoke_with_checkpointer, resume as resume_run, stream as stream_run,
};

use crate::conversions::{json_to_py, map_to_pydict, py_to_json, pydict_to_map};
use crate::py_savers::{ExtractCheckpointer, PyMemorySaver};

#[cfg(feature = "sqlite")]
use crate::py_savers::PySqliteSaver;
#[cfg(feature = "postgres")]
use crate::py_savers::PyPostgresSaver;

#[pyclass(name = "CompiledStateGraph", module = "rustakka_langgraph._native")]
pub struct PyCompiledStateGraph {
    pub(crate) inner: Arc<CompiledStateGraph>,
    pub(crate) checkpointer: Option<Arc<dyn CheckpointerHook>>,
    #[allow(dead_code)]
    pub(crate) store: Option<PyObject>,
}

impl PyCompiledStateGraph {
    pub fn from_compiled(g: CompiledStateGraph) -> Self {
        Self { inner: Arc::new(g), checkpointer: None, store: None }
    }

    pub fn attach_checkpointer(
        &mut self,
        _py: Python<'_>,
        cp: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let Some(cp) = cp else { return Ok(()) };
        // Try each known concrete type.
        if let Ok(saver) = cp.extract::<PyMemorySaver>() {
            self.checkpointer = Some(CheckpointerHookAdapter::new(saver.into_inner()));
            return Ok(());
        }
        #[cfg(feature = "sqlite")]
        if let Ok(saver) = cp.extract::<PySqliteSaver>() {
            self.checkpointer = Some(CheckpointerHookAdapter::new(saver.into_inner()));
            return Ok(());
        }
        #[cfg(feature = "postgres")]
        if let Ok(saver) = cp.extract::<PyPostgresSaver>() {
            self.checkpointer = Some(CheckpointerHookAdapter::new(saver.into_inner()));
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "checkpointer must be a MemorySaver / SqliteSaver / PostgresSaver instance",
        ))
    }

    pub fn attach_store(&mut self, _py: Python<'_>, st: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.store = st.map(|b| b.unbind());
        Ok(())
    }

    fn cfg_from(&self, cfg: Option<&Bound<'_, PyDict>>) -> PyResult<RunnableConfig> {
        let mut rc = RunnableConfig::default();
        if let Some(d) = cfg {
            if let Some(c) = d.get_item("configurable")? {
                let cd = c.downcast::<PyDict>()?;
                for (k, v) in cd.iter() {
                    rc.configurable.insert(k.extract()?, py_to_json(&v)?);
                }
            }
            if let Some(rl) = d.get_item("recursion_limit")? {
                rc.recursion_limit = Some(rl.extract()?);
            }
        }
        Ok(rc)
    }
}

#[pymethods]
impl PyCompiledStateGraph {
    #[pyo3(signature = (input, config=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
        config: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let map = if let Ok(d) = input.downcast::<PyDict>() {
            pydict_to_map(Some(d))?
        } else {
            let mut m = std::collections::BTreeMap::new();
            m.insert("input".into(), py_to_json(&input)?);
            m
        };
        let cfg = self.cfg_from(config.as_ref())?;
        let app = self.inner.clone();
        let cp = self.checkpointer.clone();
        let result = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                if let Some(cp) = cp {
                    invoke_with_checkpointer(app, map, cfg, cp).await.map(|r| r.values)
                } else {
                    invoke_dynamic(app, map, cfg).await
                }
            })
        });
        let values = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let d = map_to_pydict(py, &values)?;
        Ok(d.into_any())
    }

    #[pyo3(signature = (input, config=None))]
    fn ainvoke<'py>(
        &self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
        config: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let map = if let Ok(d) = input.downcast::<PyDict>() {
            pydict_to_map(Some(d))?
        } else {
            let mut m = std::collections::BTreeMap::new();
            m.insert("input".into(), py_to_json(&input)?);
            m
        };
        let cfg = self.cfg_from(config.as_ref())?;
        let app = self.inner.clone();
        let cp = self.checkpointer.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let res = if let Some(cp) = cp {
                invoke_with_checkpointer(app, map, cfg, cp).await.map(|r| r.values)
            } else {
                invoke_dynamic(app, map, cfg).await
            };
            let values = res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::with_gil(|py| {
                let d = map_to_pydict(py, &values)?;
                Ok::<_, PyErr>(d.into_any().unbind())
            })
        })
    }

    #[pyo3(signature = (input, config=None, stream_mode=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
        config: Option<Bound<'py, PyDict>>,
        stream_mode: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let map = pydict_to_map(input.downcast::<PyDict>().ok())?;
        let cfg = self.cfg_from(config.as_ref())?;
        let modes = match stream_mode.as_deref() {
            Some("updates") => vec![StreamMode::Updates],
            Some("messages") => vec![StreamMode::Messages],
            Some("custom") => vec![StreamMode::Custom],
            Some("debug") => vec![StreamMode::Debug],
            _ => vec![StreamMode::Values],
        };
        let app = self.inner.clone();
        let events = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                let (mut rx, _h) = stream_run(app, map, cfg, modes);
                let mut out = Vec::new();
                while let Some(ev) = rx.recv().await {
                    out.push(ev);
                }
                out
            })
        });
        let list = pyo3::types::PyList::empty_bound(py);
        for ev in events {
            let v = serde_json::to_value(&ev).unwrap_or_default();
            list.append(json_to_py(py, &v)?)?;
        }
        Ok(list.into_any())
    }

    #[pyo3(signature = (config, resume_value))]
    fn resume<'py>(
        &self,
        py: Python<'py>,
        config: Bound<'py, PyDict>,
        resume_value: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = self.cfg_from(Some(&config))?;
        let cp = self
            .checkpointer
            .clone()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("resume requires a checkpointer"))?;
        let rv = py_to_json(&resume_value)?;
        let app = self.inner.clone();
        let result = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(async move { resume_run(app, cfg, cp, rv).await })
        });
        let r = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let d = map_to_pydict(py, &r.values)?;
        Ok(d.into_any())
    }
}
