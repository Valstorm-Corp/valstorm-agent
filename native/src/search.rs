use ignore::WalkBuilder;
use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const IGNORED_NAMES: &[&str] = &[
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    ".cache",
    ".hermes",
    ".DS_Store",
];

pub fn search_files_rs(
    root_dir: &str,
    pattern: &str,
    target: &str,
    file_glob: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let root_path = Path::new(root_dir);
    if !root_path.exists() {
        return Err(format!("Directory does not exist: {}", root_dir));
    }

    let glob_matcher = if let Some(glob_str) = file_glob {
        let pattern_clean = glob_str.trim();
        let mut builder = globset::GlobBuilder::new(pattern_clean);
        builder.case_insensitive(true);
        let glob = builder.build().map_err(|e| format!("Invalid file glob '{pattern_clean}': {e}"))?;
        let mut set_builder = globset::GlobSetBuilder::new();
        set_builder.add(glob);
        Some(set_builder.build().map_err(|e| format!("Invalid glob set: {e}"))?)
    } else {
        None
    };

    let regex = Regex::new(pattern).map_err(|e| format!("Invalid regex pattern '{pattern}': {e}"))?;

    let mut builder = WalkBuilder::new(root_path);
    builder
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .filter_entry(|entry| {
            if let Some(name) = entry.file_name().to_str() {
                if IGNORED_NAMES.contains(&name) {
                    return false;
                }
            }
            true
        });

    let walker = builder.build();
    let mut matches = Vec::new();

    if target == "files" {
        for result in walker {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                continue;
            }

            let file_name = entry.file_name().to_string_lossy();
            if file_name.starts_with('.') {
                continue;
            }

            if let Some(ref gs) = glob_matcher {
                if !gs.is_match(&*file_name) {
                    continue;
                }
            }

            let rel_path = entry
                .path()
                .strip_prefix(root_path)
                .unwrap_or_else(|_| entry.path())
                .to_string_lossy();

            if regex.is_match(&file_name) || regex.is_match(&rel_path) || file_name.contains(pattern) {
                matches.push(rel_path.to_string());
                if matches.len() >= limit {
                    break;
                }
            }
        }
        return Ok(matches);
    }

    if target == "content" {
        for result in walker {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                continue;
            }

            let file_name = entry.file_name().to_string_lossy();
            if file_name.starts_with('.') {
                continue;
            }

            if let Some(ref gs) = glob_matcher {
                if !gs.is_match(&*file_name) {
                    continue;
                }
            }

            let path = entry.path();
            let file = match File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };

            let reader = BufReader::new(file);
            let rel_path = path
                .strip_prefix(root_path)
                .unwrap_or(path)
                .to_string_lossy();

            for (idx, line_res) in reader.lines().enumerate() {
                let line = match line_res {
                    Ok(l) => l,
                    Err(_) => break, // Binary or invalid UTF-8
                };

                if regex.is_match(&line) {
                    let mut clean_line = line.trim().to_string();
                    if clean_line.len() > 200 {
                        clean_line.truncate(197);
                        clean_line.push_str("...");
                    }
                    matches.push(format!("{}:{}: {}", rel_path, idx + 1, clean_line));
                    if matches.len() >= limit {
                        return Ok(matches);
                    }
                }
            }
        }
        return Ok(matches);
    }

    Err(format!("Unsupported search target '{target}'. Use 'content' or 'files'."))
}
