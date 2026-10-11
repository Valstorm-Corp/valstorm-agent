use regex::Regex;
use std::path::Path;

pub fn generate_outline_rs(file_path: &str, content: &str) -> String {
    let path = Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "py" => outline_python(content),
        "ts" | "tsx" | "js" | "jsx" => outline_typescript(content),
        "rs" => outline_rust(content),
        "go" => outline_go(content),
        _ => outline_generic(content),
    }
}

fn outline_python(content: &str) -> String {
    let mut lines = Vec::new();
    let class_re = Regex::new(r"^(class\s+[A-Za-z0-9_]+(?:\([^)]*\))?:)").unwrap();
    let def_re = Regex::new(r"^(\s*)(async\s+def|def)\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:").unwrap();
    let import_re = Regex::new(r"^(from\s+[A-Za-z0-9_.]+\s+import|import\s+[A-Za-z0-9_.]+)").unwrap();
    let const_re = Regex::new(r"^([A-Z0-9_]{3,})\s*[:=]").unwrap();

    let mut in_docstring = false;
    let mut docstring_quote = "";

    for line in content.lines() {
        let trimmed = line.trim();

        // Handle docstrings
        if !in_docstring {
            if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                let quote = &trimmed[..3];
                if trimmed.len() > 3 && trimmed.ends_with(quote) && trimmed.len() > 6 {
                    lines.push(line.to_string());
                    continue;
                }
                in_docstring = true;
                docstring_quote = quote;
                lines.push(line.to_string());
                continue;
            }
        } else {
            if trimmed.ends_with(docstring_quote) {
                in_docstring = false;
            }
            lines.push(line.to_string());
            continue;
        }

        if import_re.is_match(trimmed) {
            lines.push(line.to_string());
            continue;
        }

        if let Some(caps) = class_re.captures(trimmed) {
            lines.push(format!("\n{}", &caps[1]));
            continue;
        }

        if let Some(caps) = def_re.captures(line) {
            let indent = &caps[1];
            let kind = &caps[2];
            let name = &caps[3];
            let params = &caps[4];
            let ret = caps.get(5).map(|m| format!(" -> {}", m.as_str().trim())).unwrap_or_default();
            lines.push(format!("{}{kind} {name}({params}){ret}: ...", indent));
            continue;
        }

        if const_re.is_match(trimmed) {
            lines.push(line.to_string());
        }
    }

    if lines.is_empty() {
        return "# [No top-level classes or functions found]".to_string();
    }

    lines.join("\n")
}

fn outline_typescript(content: &str) -> String {
    let mut lines = Vec::new();
    let export_re = Regex::new(
        r"^(export\s+(?:default\s+)?(?:declare\s+)?(?:async\s+)?(?:class|function|interface|type|const|enum|let|var)\s+[^;{=]+)",
    )
    .unwrap();
    let import_re = Regex::new(r"^(import\s+[^;]+;)").unwrap();
    let method_re = Regex::new(r"^\s*(?:public|private|protected|async|static|\*)*\s*([A-Za-z0-9_]+)\s*\([^)]*\)(?:\s*:\s*[^{;]+)?").unwrap();

    let mut in_multiline_comment = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if in_multiline_comment {
            if trimmed.contains("*/") {
                in_multiline_comment = false;
            }
            continue;
        }
        if trimmed.starts_with("/*") && !trimmed.contains("*/") {
            in_multiline_comment = true;
            continue;
        }
        if trimmed.starts_with("//") {
            continue;
        }

        if import_re.is_match(trimmed) || trimmed.starts_with("import ") {
            lines.push(trimmed.to_string());
            continue;
        }

        if let Some(caps) = export_re.captures(trimmed) {
            let sig = &caps[1];
            if sig.contains("interface") || sig.contains("type") || sig.contains("enum") {
                lines.push(line.to_string());
            } else {
                lines.push(format!("{} {{ ... }}", sig));
            }
            continue;
        }

        if trimmed.starts_with("interface ") || trimmed.starts_with("type ") {
            lines.push(line.to_string());
            continue;
        }

        if let Some(caps) = method_re.captures(line) {
            if trimmed.ends_with('{') || trimmed.ends_with(';') {
                let m = caps.get(0).unwrap().as_str().trim_end_matches('{').trim();
                lines.push(format!("  {};", m));
            }
        }
    }

    if lines.is_empty() {
        return "// [No exports or types found]".to_string();
    }

    lines.join("\n")
}

fn outline_rust(content: &str) -> String {
    let mut lines = Vec::new();
    let pub_re = Regex::new(r"^(pub(\([^)]+\))?\s+(fn|struct|enum|trait|type|const|static)\s+[^{;]+)").unwrap();
    let impl_re = Regex::new(r"^(impl\s+[^{]+)").unwrap();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("use ") {
            lines.push(trimmed.to_string());
            continue;
        }
        if let Some(caps) = pub_re.captures(trimmed) {
            lines.push(format!("{};", &caps[1]));
            continue;
        }
        if let Some(caps) = impl_re.captures(trimmed) {
            lines.push(format!("{} {{ ... }}", &caps[1]));
        }
    }

    lines.join("\n")
}

fn outline_go(content: &str) -> String {
    let mut lines = Vec::new();
    let type_re = Regex::new(r"^(type\s+[A-Za-z0-9_]+\s+(struct|interface))").unwrap();
    let func_re = Regex::new(r"^(func\s+(\([^)]+\)\s+)?[A-Za-z0-9_]+\s*\([^)]*\)\s*([^{]+)?)").unwrap();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("package ") || trimmed.starts_with("import ") {
            lines.push(trimmed.to_string());
            continue;
        }
        if let Some(caps) = type_re.captures(trimmed) {
            lines.push(format!("{} {{ ... }}", &caps[1]));
            continue;
        }
        if let Some(caps) = func_re.captures(trimmed) {
            lines.push(format!("{} {{ ... }}", &caps[1]));
        }
    }

    lines.join("\n")
}

fn outline_generic(content: &str) -> String {
    let lines: Vec<&str> = content.lines().take(40).collect();
    lines.join("\n")
}
