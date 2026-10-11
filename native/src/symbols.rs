use ignore::WalkBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SymbolMatch {
    pub file: String,
    pub line: usize,
    pub kind: String,
    pub name: String,
    pub signature: String,
}

const SUPPORTED_EXTS: &[&str] = &["py", "ts", "tsx", "js", "jsx", "rs", "go"];
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
    ".next",
    ".turbo",
    ".cache",
];

pub fn find_symbols_rs(
    root_dir: &str,
    query: &str,
    symbol_kind: Option<&str>,
    limit: usize,
) -> Result<Vec<SymbolMatch>, String> {
    let root_path = Path::new(root_dir);
    if !root_path.exists() {
        return Err(format!("Directory does not exist: {}", root_dir));
    }

    let query_lower = query.to_lowercase();
    let query_regex = Regex::new(&format!(r"(?i)\b{}\b", regex::escape(query)))
        .unwrap_or_else(|_| Regex::new(&regex::escape(query)).unwrap());

    // Definition patterns:
    // Python: def foo / class Foo
    let py_def = Regex::new(r"^(?:\s*)(class|(?:async\s+)?def)\s+([A-Za-z0-9_]+)").unwrap();
    // TS/JS: (export)? (class|interface|type|enum|function|const|let) Foo
    let ts_def = Regex::new(
        r"(?:export\s+(?:default\s+)?)?(class|interface|type|enum|function|const|let|var)\s+([A-Za-z0-9_]+)",
    )
    .unwrap();
    // Rust: pub (fn|struct|enum|trait|type|const) Foo
    let rs_def = Regex::new(r"(?:pub(?:\([^)]+\))?\s+)?(fn|struct|enum|trait|type|const)\s+([A-Za-z0-9_]+)").unwrap();
    // Go: func (r receiver)? Foo / type Foo struct/interface
    let go_def = Regex::new(r"(func|type)\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)").unwrap();

    let mut builder = WalkBuilder::new(root_path);
    builder
        .hidden(true)
        .git_ignore(true)
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

    for result in walker {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map_or(false, |ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => continue,
        };
        if !SUPPORTED_EXTS.contains(&ext.as_str()) {
            continue;
        }

        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let reader = BufReader::new(file);
        let rel_path = path
            .strip_prefix(root_path)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        for (idx, line_res) in reader.lines().enumerate() {
            let line = match line_res {
                Ok(l) => l,
                Err(_) => break,
            };

            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
                continue;
            }

            let parsed: Option<(&str, &str)> = match ext.as_str() {
                "py" => py_def.captures(trimmed).map(|c| {
                    let k = if c[1].contains("def") { "function" } else { "class" };
                    (k, c.get(2).unwrap().as_str())
                }),
                "ts" | "tsx" | "js" | "jsx" => ts_def.captures(trimmed).map(|c| {
                    let k = match &c[1] {
                        "function" => "function",
                        "class" => "class",
                        "interface" => "interface",
                        "type" => "type",
                        "enum" => "enum",
                        _ => "variable",
                    };
                    (k, c.get(2).unwrap().as_str())
                }),
                "rs" => rs_def.captures(trimmed).map(|c| {
                    let k = match &c[1] {
                        "fn" => "function",
                        "struct" => "struct",
                        "enum" => "enum",
                        "trait" => "trait",
                        "type" => "type",
                        _ => "constant",
                    };
                    (k, c.get(2).unwrap().as_str())
                }),
                "go" => go_def.captures(trimmed).map(|c| {
                    let k = if &c[1] == "func" { "function" } else { "type" };
                    (k, c.get(2).unwrap().as_str())
                }),
                _ => None,
            };

            if let Some((kind, name)) = parsed {
                if let Some(target_kind) = symbol_kind {
                    if !kind.eq_ignore_ascii_case(target_kind) {
                        continue;
                    }
                }

                let matches_query = if query.is_empty() {
                    true
                } else if query_regex.is_match(name) || name.to_lowercase().contains(&query_lower) {
                    true
                } else {
                    false
                };

                if matches_query {
                    let mut sig = trimmed.to_string();
                    if sig.len() > 160 {
                        sig.truncate(157);
                        sig.push_str("...");
                    }
                    matches.push(SymbolMatch {
                        file: rel_path.clone(),
                        line: idx + 1,
                        kind: kind.to_string(),
                        name: name.to_string(),
                        signature: sig,
                    });

                    if matches.len() >= limit {
                        return Ok(matches);
                    }
                }
            }
        }
    }

    Ok(matches)
}
