//! `PyCompiledStateGraph` — runtime handle exposing `invoke`/`ainvoke`/
//! `stream`/`astream`/`get_state`/`update_state`/`with_config`.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use rustakka_langgraph_checkpoint::base::Checkpointer;
use rustakka_langgraph_checkpoint::CheckpointerHookAdapter;
use rustakka_langgraph_core::config::{RunnableConfig, StreamMode};
use rustakka_langgraph_core::context::StoreAccessor;
use rustakka_langgraph_core::coordinator::CheckpointerHook;
use rustakka_langgraph_core::graph::CompiledStateGraph;
use rustakka_langgraph_core::runner::{
    get_state as runner_get_state, get_state_history as runner_get_state_history, invoke_dynamic,
    invoke_with_checkpointer, invoke_with_store, resume as resume_run, stream as stream_run,
    update_state as runner_update_state, StateSnapshot,
};
use rustakka_langgraph_store::store_accessor;

use crate::py_stores::PyInMemoryStore;
#[cfg(feature = "postgres")]
use crate::py_stores::PyPostgresStore;

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
    /// Concrete `StoreAccessor` extracted from `attach_store`, threaded
    /// into every run so nodes can reach the store via
    /// [`rustakka_langgraph_core::context::get_store`].
    pub(crate) store_accessor: Option<Arc<dyn StoreAccessor>>,
    /// Default `RunnableConfig` merged into every call that accepts a
    /// `config=` kwarg. Populated by [`PyCompiledStateGraph::with_config`].
    pub(crate) default_cfg: RunnableConfig,
}

impl PyCompiledStateGraph {
    pub fn from_compiled(g: CompiledStateGraph) -> Self {
        Self {
            inner: Arc::new(g),
            checkpointer: None,
            store: None,
            store_accessor: None,
            default_cfg: RunnableConfig::default(),
        }
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
        let Some(st) = st else {
            self.store = None;
            self.store_accessor = None;
            return Ok(());
        };
        // Try each known concrete store type; on success, stash both the
        // `PyObject` (for introspection) and a `StoreAccessor` the core
        // engine can thread into node executions.
        if let Ok(ims) = st.extract::<PyInMemoryStore>() {
            self.store_accessor = Some(store_accessor(ims.inner.clone()));
            self.store = Some(st.unbind());
            return Ok(());
        }
        #[cfg(feature = "postgres")]
        if let Ok(pgs) = st.extract::<PyPostgresStore>() {
            self.store_accessor = Some(store_accessor(pgs.inner.clone()));
            self.store = Some(st.unbind());
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "store must be an InMemoryStore / PostgresStore instance",
        ))
    }

    fn cfg_from(&self, cfg: Option<&Bound<'_, PyDict>>) -> PyResult<RunnableConfig> {
        let mut rc = self.default_cfg.clone();
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
            if let Some(tid) = d.get_item("thread_id")? {
                rc.configurable.insert("thread_id".into(), py_to_json(&tid)?);
            }
            if let Some(ns) = d.get_item("checkpoint_ns")? {
                rc.configurable.insert("checkpoint_ns".into(), py_to_json(&ns)?);
            }
            if let Some(cid) = d.get_item("checkpoint_id")? {
                rc.checkpoint_id = Some(cid.extract()?);
            }
        }
        Ok(rc)
    }
}

fn parse_stream_modes(v: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<StreamMode>> {
    let Some(v) = v else { return Ok(vec![StreamMode::Values]) };
    if v.is_none() {
        return Ok(vec![StreamMode::Values]);
    }
    if let Ok(s) = v.extract::<String>() {
        let m = StreamMode::parse(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown stream_mode `{s}`"))
        })?;
        return Ok(vec![m]);
    }
    if let Ok(list) = v.extract::<Vec<String>>() {
        let mut modes = Vec::with_capacity(list.len());
        for s in list {
            let m = StreamMode::parse(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("unknown stream_mode `{s}`"))
            })?;
            modes.push(m);
        }
        return Ok(modes);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "stream_mode must be a string or list of strings",
    ))
}

fn snapshot_to_pydict<'py>(py: Python<'py>, s: &StateSnapshot) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("values", map_to_pydict(py, &s.values)?)?;
    d.set_item("step", s.step)?;
    match &s.interrupt {
        Some(intr) => d.set_item(
            "interrupt",
            json_to_py(py, &serde_json::to_value(intr).unwrap_or_default())?,
        )?,
        None => d.set_item("interrupt", py.None())?,
    }
    let cfg_d = PyDict::new_bound(py);
    if let Some(tid) = s.config.thread_id() {
        cfg_d.set_item("thread_id", tid)?;
    }
    cfg_d.set_item("checkpoint_ns", s.config.checkpoint_ns())?;
    if let Some(cid) = &s.config.checkpoint_id {
        cfg_d.set_item("checkpoint_id", cid)?;
    }
    d.set_item("config", cfg_d)?;
    Ok(d)
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
        let store = self.store_accessor.clone();
        let result = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                match (cp, store) {
                    (Some(cp), _) => {
                        invoke_with_checkpointer(app, map, cfg, cp).await.map(|r| r.values)
                    }
                    (None, Some(s)) => {
                        invoke_with_store(app, map, cfg, None, s).await.map(|r| r.values)
                    }
                    (None, None) => invoke_dynamic(app, map, cfg).await,
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
        let store = self.store_accessor.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let res = match (cp, store) {
                (Some(cp), _) => {
                    invoke_with_checkpointer(app, map, cfg, cp).await.map(|r| r.values)
                }
                (None, Some(s)) => {
                    invoke_with_store(app, map, cfg, None, s).await.map(|r| r.values)
                }
                (None, None) => invoke_dynamic(app, map, cfg).await,
            };
            let values = res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::with_gil(|py| {
                let d = map_to_pydict(py, &values)?;
                Ok::<_, PyErr>(d.into_any().unbind())
            })
        })
    }

    #[pyo3(signature = (input, config=None, stream_mode=None, subgraphs=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
        config: Option<Bound<'py, PyDict>>,
        stream_mode: Option<Bound<'py, PyAny>>,
        subgraphs: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let map = pydict_to_map(input.downcast::<PyDict>().ok())?;
        let cfg = self.cfg_from(config.as_ref())?;
        let modes = parse_stream_modes(stream_mode.as_ref())?;
        let multi_mode = modes.len() > 1;
        let want_subgraphs = subgraphs.unwrap_or(false);
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
            // Drop subgraph-nested events unless the caller opted in.
            let ns: &Vec<String> = match &ev {
                rustakka_langgraph_core::stream::StreamEvent::Values { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::Updates { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::Messages { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::Custom { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::Debug { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::OnChainStart { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::OnChainEnd { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::OnChatModelStream { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::OnToolStart { namespace, .. }
                | rustakka_langgraph_core::stream::StreamEvent::OnToolEnd { namespace, .. } => {
                    namespace
                }
            };
            if !want_subgraphs && !ns.is_empty() {
                continue;
            }
            let v = serde_json::to_value(&ev).unwrap_or_default();
            let payload = json_to_py(py, &v)?;
            // Shape matches upstream:
            //   - single mode, no subgraphs: bare event dict
            //   - multi mode:                 ("mode", event_dict)
            //   - subgraphs=True:             (namespace_tuple, event_dict) or
            //                                 (ns, "mode", event_dict) if multi
            let shaped: Bound<'py, PyAny> = if want_subgraphs && multi_mode {
                let ns_tup = pyo3::types::PyTuple::new_bound(py, ns);
                let mode_str = ev.mode_string();
                pyo3::types::PyTuple::new_bound(
                    py,
                    &[ns_tup.into_any(), mode_str.to_object(py).into_bound(py), payload],
                )
                .into_any()
            } else if want_subgraphs {
                let ns_tup = pyo3::types::PyTuple::new_bound(py, ns);
                pyo3::types::PyTuple::new_bound(py, &[ns_tup.into_any(), payload]).into_any()
            } else if multi_mode {
                let mode_str = ev.mode_string();
                pyo3::types::PyTuple::new_bound(
                    py,
                    &[mode_str.to_object(py).into_bound(py), payload],
                )
                .into_any()
            } else {
                payload
            };
            list.append(shaped)?;
        }
        Ok(list.into_any())
    }

    /// Return a **new** wrapper that applies `config` as the default for
    /// every subsequent call. Mirrors upstream `.with_config(...)`.
    #[pyo3(signature = (config))]
    fn with_config(&self, config: Bound<'_, PyDict>) -> PyResult<Self> {
        let extra = self.cfg_from(Some(&config))?;
        let mut merged = self.default_cfg.clone();
        for (k, v) in extra.configurable {
            merged.configurable.insert(k, v);
        }
        if extra.recursion_limit.is_some() {
            merged.recursion_limit = extra.recursion_limit;
        }
        if extra.checkpoint_id.is_some() {
            merged.checkpoint_id = extra.checkpoint_id;
        }
        Ok(Self {
            inner: self.inner.clone(),
            checkpointer: self.checkpointer.clone(),
            store: None,
            store_accessor: self.store_accessor.clone(),
            default_cfg: merged,
        })
    }

    /// Fetch the latest (or `cfg["checkpoint_id"]`-specific) state snapshot.
    /// Returns a dict with `values`, `step`, `interrupt`, and `config`.
    #[pyo3(signature = (config=None))]
    fn get_state<'py>(
        &self,
        py: Python<'py>,
        config: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = self.cfg_from(config.as_ref())?;
        let cp = self.checkpointer.clone().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("get_state requires a checkpointer")
        })?;
        let app = self.inner.clone();
        let res = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(async move { runner_get_state(&app, &cfg, cp).await })
        });
        let snap = res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        match snap {
            None => Ok(py.None().into_bound(py)),
            Some(s) => Ok(snapshot_to_pydict(py, &s)?.into_any()),
        }
    }

    /// List prior checkpoints (newest first). Mirrors
    /// `CompiledStateGraph.get_state_history(config, limit=None)`.
    #[pyo3(signature = (config=None, limit=None))]
    fn get_state_history<'py>(
        &self,
        py: Python<'py>,
        config: Option<Bound<'py, PyDict>>,
        limit: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = self.cfg_from(config.as_ref())?;
        let cp = self.checkpointer.clone().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("get_state_history requires a checkpointer")
        })?;
        let app = self.inner.clone();
        let res = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                runner_get_state_history(&app, &cfg, cp, limit).await
            })
        });
        let snaps = res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty_bound(py);
        for s in &snaps {
            list.append(snapshot_to_pydict(py, s)?)?;
        }
        Ok(list.into_any())
    }

    /// Patch thread state without running a node. Mirrors
    /// `CompiledStateGraph.update_state(config, values, as_node=None)`.
    #[pyo3(signature = (config, values, as_node=None))]
    fn update_state<'py>(
        &self,
        py: Python<'py>,
        config: Bound<'py, PyDict>,
        values: Bound<'py, PyDict>,
        as_node: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = self.cfg_from(Some(&config))?;
        let map = pydict_to_map(Some(&values))?;
        let cp = self.checkpointer.clone().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("update_state requires a checkpointer")
        })?;
        let app = self.inner.clone();
        let res = py.allow_threads(|| {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                runner_update_state(&app, &cfg, cp, map, as_node).await
            })
        });
        let new_cfg = res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let d = PyDict::new_bound(py);
        if let Some(tid) = new_cfg.thread_id() {
            d.set_item("thread_id", tid)?;
        }
        d.set_item("checkpoint_ns", new_cfg.checkpoint_ns())?;
        if let Some(cid) = &new_cfg.checkpoint_id {
            d.set_item("checkpoint_id", cid)?;
        }
        Ok(d.into_any())
    }

    /// Render this graph as a Mermaid `flowchart TD` diagram.
    fn draw_mermaid(&self) -> String {
        self.inner.draw_mermaid()
    }

    /// Render a minimal ASCII overview (nodes + edges).
    fn draw_ascii(&self) -> String {
        self.inner.draw_ascii()
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
