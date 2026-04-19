//! Python wrappers for checkpoint savers.

use std::sync::Arc;

use pyo3::prelude::*;

use rustakka_langgraph_checkpoint::base::Checkpointer;
use rustakka_langgraph_checkpoint::MemorySaver;

pub trait ExtractCheckpointer {
    type Inner: Checkpointer;
    fn into_inner(self) -> Arc<Self::Inner>;
}

#[pyclass(name = "MemorySaver", module = "rustakka_langgraph._native")]
#[derive(Clone)]
pub struct PyMemorySaver {
    inner: Arc<MemorySaver>,
}

#[pymethods]
impl PyMemorySaver {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(MemorySaver::new()) }
    }
}

impl ExtractCheckpointer for PyMemorySaver {
    type Inner = MemorySaver;
    fn into_inner(self) -> Arc<MemorySaver> {
        self.inner
    }
}

#[cfg(feature = "sqlite")]
mod sqlite_impl {
    use super::*;
    use rustakka_langgraph::checkpoint_sqlite::SqliteSaver;

    #[pyclass(name = "SqliteSaver", module = "rustakka_langgraph._native")]
    #[derive(Clone)]
    pub struct PySqliteSaver {
        pub(crate) inner: Arc<SqliteSaver>,
    }

    #[pymethods]
    impl PySqliteSaver {
        #[staticmethod]
        fn from_url<'py>(py: Python<'py>, url: String) -> PyResult<Bound<'py, PyAny>> {
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let saver = SqliteSaver::from_url(&url)
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Python::with_gil(|py| {
                    Ok::<_, PyErr>(
                        Py::new(py, PySqliteSaver { inner: Arc::new(saver) })?
                            .into_bound(py)
                            .into_any()
                            .unbind(),
                    )
                })
            })
        }

        #[staticmethod]
        fn from_url_blocking(url: String) -> PyResult<Self> {
            let inner = pyo3_async_runtimes::tokio::get_runtime()
                .block_on(async move { SqliteSaver::from_url(&url).await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner: Arc::new(inner) })
        }
    }

    impl ExtractCheckpointer for PySqliteSaver {
        type Inner = SqliteSaver;
        fn into_inner(self) -> Arc<SqliteSaver> {
            self.inner
        }
    }
}

#[cfg(feature = "sqlite")]
pub use sqlite_impl::PySqliteSaver;

#[cfg(feature = "postgres")]
mod postgres_impl {
    use super::*;
    use rustakka_langgraph::checkpoint_postgres::PostgresSaver;

    #[pyclass(name = "PostgresSaver", module = "rustakka_langgraph._native")]
    #[derive(Clone)]
    pub struct PyPostgresSaver {
        pub(crate) inner: Arc<PostgresSaver>,
    }

    #[pymethods]
    impl PyPostgresSaver {
        #[staticmethod]
        #[pyo3(signature = (url, schema=None))]
        fn from_url_blocking(url: String, schema: Option<String>) -> PyResult<Self> {
            let inner = pyo3_async_runtimes::tokio::get_runtime()
                .block_on(async move {
                    PostgresSaver::from_url_with_schema(&url, schema.as_deref().unwrap_or("public")).await
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner: Arc::new(inner) })
        }
    }

    impl ExtractCheckpointer for PyPostgresSaver {
        type Inner = PostgresSaver;
        fn into_inner(self) -> Arc<PostgresSaver> {
            self.inner
        }
    }
}

#[cfg(feature = "postgres")]
pub use postgres_impl::PyPostgresSaver;
