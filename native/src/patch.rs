use similar::{ChangeTag, TextDiff};

pub struct PatchResult {
    pub patched_content: String,
    pub diff: String,
}

pub fn patch_content_rs(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<PatchResult, String> {
    if old_string.is_empty() {
        return Err("The 'old_string' parameter cannot be empty.".to_string());
    }

    // 1. Try direct exact match
    let count = content.matches(old_string).count();

    if count == 1 || (count > 1 && replace_all) {
        let patched = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };
        let diff = generate_unified_diff(content, &patched, "original", "patched");
        return Ok(PatchResult {
            patched_content: patched,
            diff,
        });
    }

    if count > 1 && !replace_all {
        return Err(format!(
            "Found {} occurrences of 'old_string' in file. Set replace_all=True or provide more surrounding context to match uniquely.",
            count
        ));
    }

    // 2. Try normalized newline match (\r\n vs \n)
    let norm_content = content.replace("\r\n", "\n");
    let norm_old = old_string.replace("\r\n", "\n");
    let norm_new = new_string.replace("\r\n", "\n");

    let norm_count = norm_content.matches(&norm_old).count();
    if norm_count == 1 || (norm_count > 1 && replace_all) {
        let is_original_crlf = content.contains("\r\n");
        let patched_norm = if replace_all {
            norm_content.replace(&norm_old, &norm_new)
        } else {
            norm_content.replacen(&norm_old, &norm_new, 1)
        };
        let patched = if is_original_crlf {
            patched_norm.replace('\n', "\r\n")
        } else {
            patched_norm
        };
        let diff = generate_unified_diff(content, &patched, "original", "patched");
        return Ok(PatchResult {
            patched_content: patched,
            diff,
        });
    }

    // 3. Line-trimmed fuzzy fallback match
    let old_lines: Vec<&str> = norm_old.lines().map(|l| l.trim()).collect();
    if !old_lines.is_empty() {
        let file_lines: Vec<&str> = norm_content.lines().collect();
        let old_len = old_lines.len();

        let mut candidate_indices = Vec::new();
        for i in 0..=file_lines.len().saturating_sub(old_len) {
            let slice = &file_lines[i..i + old_len];
            let matches = slice
                .iter()
                .zip(old_lines.iter())
                .all(|(fl, ol)| fl.trim() == *ol);
            if matches {
                candidate_indices.push(i);
            }
        }

        if candidate_indices.len() == 1 {
            let start_idx = candidate_indices[0];
            let mut result_lines = Vec::new();

            // lines before match
            for line in &file_lines[..start_idx] {
                result_lines.push(line.to_string());
            }

            // determine base indent of the first matched line
            let original_first_line = file_lines[start_idx];
            let base_indent: String = original_first_line
                .chars()
                .take_while(|c| c.is_whitespace())
                .collect();

            let new_lines: Vec<&str> = norm_new.lines().collect();
            for (idx, nl) in new_lines.iter().enumerate() {
                if idx > 0 && !nl.starts_with(&base_indent) && !nl.trim().is_empty() {
                    result_lines.push(format!("{}{}", base_indent, nl.trim_start()));
                } else {
                    result_lines.push(nl.to_string());
                }
            }

            // lines after match
            for line in &file_lines[start_idx + old_len..] {
                result_lines.push(line.to_string());
            }

            let mut patched = result_lines.join("\n");
            if norm_content.ends_with('\n') && !patched.ends_with('\n') {
                patched.push('\n');
            }
            if content.contains("\r\n") {
                patched = patched.replace('\n', "\r\n");
            }

            let diff = generate_unified_diff(content, &patched, "original", "patched");
            return Ok(PatchResult {
                patched_content: patched,
                diff,
            });
        } else if candidate_indices.len() > 1 && !replace_all {
            return Err(format!(
                "Found {} fuzzy matches for old_string. Please provide more surrounding lines for a unique match.",
                candidate_indices.len()
            ));
        }
    }

    Err(
        "Could not find exact or fuzzy match for 'old_string' in file. Ensure exact code or surrounding lines are provided."
            .to_string(),
    )
}

pub fn generate_unified_diff(original: &str, modified: &str, old_label: &str, new_label: &str) -> String {
    let diff = TextDiff::from_lines(original, modified);
    let mut output = String::new();

    for (idx, group) in diff.grouped_ops(3).iter().enumerate() {
        if idx > 0 {
            output.push_str("\n");
        }
        for op in group {
            for change in diff.iter_changes(op) {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                output.push_str(&format!("{}{}", sign, change.value()));
            }
        }
    }

    if output.is_empty() {
        return format!("--- {}\n+++ {}\n (no differences)", old_label, new_label);
    }

    format!("--- {}\n+++ {}\n{}", old_label, new_label, output)
}
