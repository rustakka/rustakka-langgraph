//! Tokio runtime bridge for `pyo3_async_runtimes`.

use once_cell::sync::OnceCell;
use pyo3::prelude::*;

static INIT: OnceCell<()> = OnceCell::new();

/// Initialize the shared tokio runtime used by `pyo3-async-runtimes` and the
/// rustakka actor system.
pub fn init(_py: Python<'_>) {
    INIT.get_or_init(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        pyo3_async_runtimes::tokio::init_with_runtime(Box::leak(Box::new(rt)))
            .expect("pyo3 async runtime init");
    });
}
