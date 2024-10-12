#![feature(iter_map_windows)]

use pyo3::prelude::*;

pub mod loc_1d;

#[pymodule]
fn _evaluation(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(loc_1d::ap_1d, m)?)?;
    m.add_function(wrap_pyfunction!(loc_1d::ar_1d, m)?)?;
    m.add_function(wrap_pyfunction!(loc_1d::ap_ar_1d, m)?)?;

    Ok(())
}
