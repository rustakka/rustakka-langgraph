//! Python wrappers for `BaseStore` implementations.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use rustakka_langgraph_store::base::{BaseStore, ListNamespacesFilter, PutOptions};
use rustakka_langgraph_store::InMemoryStore;

use crate::conversions::{json_to_py, py_to_json};

#[pyclass(name = "InMemoryStore", module = "rustakka_langgraph._native")]
#[derive(Clone)]
pub struct PyInMemoryStore {
    pub(crate) inner: Arc<InMemoryStore>,
}

#[pymethods]
impl PyInMemoryStore {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(InMemoryStore::new()) }
    }

    fn put(
        &self,
        namespace: Vec<String>,
        key: String,
        value: Bound<'_, PyAny>,
        ttl_seconds: Option<u64>,
    ) -> PyResult<()> {
        let v = py_to_json(&value)?;
        let opts = PutOptions { ttl_seconds, ..Default::default() };
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(async move { inner.put(&namespace, &key, v, opts).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn get<'py>(
        &self,
        py: Python<'py>,
        namespace: Vec<String>,
        key: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let opt = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(async move { inner.get(&namespace, &key).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        match opt {
            None => Ok(py.None().into_bound(py)),
            Some(item) => {
                let d = PyDict::new_bound(py);
                d.set_item("namespace", item.namespace)?;
                d.set_item("key", item.key)?;
                d.set_item("value", json_to_py(py, &item.value)?)?;
                d.set_item("created_at", item.created_at.to_rfc3339())?;
                d.set_item("updated_at", item.updated_at.to_rfc3339())?;
                Ok(d.into_any())
            }
        }
    }

    fn delete(&self, namespace: Vec<String>, key: String) -> PyResult<()> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(async move { inner.delete(&namespace, &key).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (namespace_prefix, query=None, limit=10, offset=0))]
    fn search(
        &self,
        py: Python<'_>,
        namespace_prefix: Vec<String>,
        query: Option<String>,
        limit: u32,
        offset: u32,
    ) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let hits = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(async move {
                inner.search(&namespace_prefix, query.as_deref(), limit, offset).await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty_bound(py);
        for h in hits {
            let d = PyDict::new_bound(py);
            d.set_item("namespace", h.item.namespace)?;
            d.set_item("key", h.item.key)?;
            d.set_item("value", json_to_py(py, &h.item.value)?)?;
            list.append(d)?;
        }
        Ok(list.into_any().unbind())
    }

    #[pyo3(signature = (prefix=None, max_depth=None, limit=None))]
    fn list_namespaces(
        &self,
        py: Python<'_>,
        prefix: Option<Vec<String>>,
        max_depth: Option<u32>,
        limit: Option<u32>,
    ) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let result = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(async move {
                inner
                    .list_namespaces(ListNamespacesFilter { prefix, max_depth, limit })
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty_bound(py);
        for ns in result {
            list.append(ns)?;
        }
        Ok(list.into_any().unbind())
    }
}

#[cfg(feature = "postgres")]
mod pg {
    use super::*;
    use rustakka_langgraph::store_postgres::PostgresStore;

    #[pyclass(name = "PostgresStore", module = "rustakka_langgraph._native")]
    #[derive(Clone)]
    pub struct PyPostgresStore {
        pub(crate) inner: Arc<PostgresStore>,
    }

    #[pymethods]
    impl PyPostgresStore {
        #[staticmethod]
        #[pyo3(signature = (url, schema=None))]
        fn from_url_blocking(url: String, schema: Option<String>) -> PyResult<Self> {
            let inner = pyo3_async_runtimes::tokio::get_runtime()
                .block_on(async move {
                    PostgresStore::from_url_with_schema(&url, schema.as_deref().unwrap_or("public")).await
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner: Arc::new(inner) })
        }
    }
}

#[cfg(feature = "postgres")]
pub use pg::PyPostgresStore;
