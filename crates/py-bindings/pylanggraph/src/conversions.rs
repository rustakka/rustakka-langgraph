//! Conversions between Python objects and `serde_json::Value`.
//!
//! We intentionally use JSON as the wire format on the FFI boundary so:
//!   - the engine never touches the GIL during message passing,
//!   - checkpoints are byte-for-byte interchangeable with upstream Python,
//!   - reducer logic remains a single source of truth in Rust.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use serde_json::Value;

pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(tup) = obj.downcast::<PyTuple>() {
        let mut out = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut out = serde_json::Map::with_capacity(d.len());
        for (k, v) in d.iter() {
            let k_s: String = k.extract()?;
            out.insert(k_s, py_to_json(&v)?);
        }
        return Ok(Value::Object(out));
    }
    // Fallback via __dict__ / __repr__
    if let Ok(s) = obj.str() {
        return Ok(Value::String(s.to_string()));
    }
    Err(PyValueError::new_err("unsupported python type for graph state"))
}

pub fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let m = match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.to_object(py).into_bound(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py).into_bound(py)
            } else if let Some(f) = n.as_f64() {
                f.to_object(py).into_bound(py)
            } else {
                py.None().into_bound(py)
            }
        }
        Value::String(s) => s.to_object(py).into_bound(py),
        Value::Array(a) => {
            let list = PyList::empty_bound(py);
            for v in a {
                list.append(json_to_py(py, v)?)?;
            }
            list.into_any()
        }
        Value::Object(o) => {
            let d = PyDict::new_bound(py);
            for (k, v) in o {
                d.set_item(k, json_to_py(py, v)?)?;
            }
            d.into_any()
        }
    };
    Ok(m)
}

pub fn pydict_to_map(d: Option<&Bound<'_, PyDict>>) -> PyResult<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    if let Some(d) = d {
        for (k, v) in d.iter() {
            out.insert(k.extract()?, py_to_json(&v)?);
        }
    }
    Ok(out)
}

pub fn map_to_pydict<'py>(
    py: Python<'py>,
    map: &BTreeMap<String, Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    for (k, v) in map {
        d.set_item(k, json_to_py(py, v)?)?;
    }
    Ok(d)
}
