use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod outline;
mod patch;
mod sanitizer;
mod search;
mod symbols;
mod topology;

#[pyfunction]
fn ping() -> &'static str {
    "vsagent-native rust core v0.1.0 operational"
}

#[pyfunction]
#[pyo3(signature = (root_dir, pattern, target="content", file_glob=None, limit=50))]
fn search_files(
    root_dir: &str,
    pattern: &str,
    target: &str,
    file_glob: Option<&str>,
    limit: usize,
) -> PyResult<Vec<String>> {
    search::search_files_rs(root_dir, pattern, target, file_glob, limit)
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (content, old_string, new_string, replace_all=false))]
fn patch_content(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> PyResult<(String, String)> {
    let res = patch::patch_content_rs(content, old_string, new_string, replace_all)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok((res.patched_content, res.diff))
}

#[pyfunction]
fn generate_outline(file_path: &str, content: &str) -> PyResult<String> {
    Ok(outline::generate_outline_rs(file_path, content))
}

#[pyfunction]
#[pyo3(signature = (root_dir, query="", kind=None, limit=50))]
fn find_symbols<'py>(
    py: Python<'py>,
    root_dir: &str,
    query: &str,
    kind: Option<&str>,
    limit: usize,
) -> PyResult<Bound<'py, PyList>> {
    let matches = symbols::find_symbols_rs(root_dir, query, kind, limit)
        .map_err(|e| PyValueError::new_err(e))?;

    let list = PyList::empty(py);
    for m in matches {
        let dict = PyDict::new(py);
        dict.set_item("file", m.file)?;
        dict.set_item("line", m.line)?;
        dict.set_item("kind", m.kind)?;
        dict.set_item("name", m.name)?;
        dict.set_item("signature", m.signature)?;
        list.append(dict)?;
    }

    Ok(list)
}

#[pyfunction]
#[pyo3(signature = (root_dir="."))]
fn scan_workspace(root_dir: &str) -> PyResult<String> {
    let report = topology::scan_workspace_rs(root_dir);
    serde_json::to_string_pretty(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn get_blast_radius(root_dir: &str, target_file: &str) -> PyResult<String> {
    let report = topology::get_blast_radius_rs(root_dir, target_file);
    serde_json::to_string_pretty(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn fast_sanitize_text(text: &str) -> PyResult<String> {
    Ok(sanitizer::fast_sanitize_text_rs(text))
}

#[pymodule]
fn vsagent_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(search_files, m)?)?;
    m.add_function(wrap_pyfunction!(patch_content, m)?)?;
    m.add_function(wrap_pyfunction!(generate_outline, m)?)?;
    m.add_function(wrap_pyfunction!(find_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(scan_workspace, m)?)?;
    m.add_function(wrap_pyfunction!(get_blast_radius, m)?)?;
    m.add_function(wrap_pyfunction!(fast_sanitize_text, m)?)?;
    Ok(())
}
