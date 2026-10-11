use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PackageInfo {
    pub name: String,
    pub path: String,
    pub dependencies: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkspaceReport {
    pub workspace_type: String,
    pub total_packages: usize,
    pub packages: Vec<PackageInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlastRadiusReport {
    pub file_path: String,
    pub owning_package: Option<String>,
    pub direct_dependents: Vec<String>,
    pub affected_package_paths: Vec<String>,
}

pub fn scan_workspace_rs(root_dir: &str) -> WorkspaceReport {
    let root = Path::new(root_dir);
    let mut packages = Vec::new();
    let mut ws_type = "unknown".to_string();

    // 1. Check package.json workspaces
    let root_pkg_json = root.join("package.json");
    if root_pkg_json.exists() {
        if let Ok(content) = fs::read_to_string(&root_pkg_json) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                let mut globs: Vec<String> = Vec::new();
                if let Some(ws) = val.get("workspaces") {
                    if let Some(arr) = ws.as_array() {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                globs.push(s.to_string());
                            }
                        }
                    } else if let Some(packages_arr) = ws.get("packages").and_then(|p| p.as_array()) {
                        for item in packages_arr {
                            if let Some(s) = item.as_str() {
                                globs.push(s.to_string());
                            }
                        }
                    }
                }

                if !globs.is_empty() {
                    ws_type = "npm/yarn/pnpm".to_string();
                    for pattern in globs {
                        let clean_glob = pattern.trim_end_matches("/*").trim_end_matches("/**");
                        let search_dir = root.join(clean_glob);
                        if search_dir.is_dir() {
                            if let Ok(entries) = fs::read_dir(search_dir) {
                                for entry in entries.flatten() {
                                    let sub_pkg_json = entry.path().join("package.json");
                                    if sub_pkg_json.exists() {
                                        if let Ok(sub_content) = fs::read_to_string(&sub_pkg_json) {
                                            if let Ok(sub_val) = serde_json::from_str::<serde_json::Value>(&sub_content) {
                                                let name = sub_val.get("name").and_then(|n| n.as_str()).unwrap_or("unnamed");
                                                let mut deps = Vec::new();
                                                for dep_field in &["dependencies", "devDependencies", "peerDependencies"] {
                                                    if let Some(obj) = sub_val.get(dep_field).and_then(|d| d.as_object()) {
                                                        for k in obj.keys() {
                                                            deps.push(k.clone());
                                                        }
                                                    }
                                                }
                                                let rel_path = entry
                                                    .path()
                                                    .strip_prefix(root)
                                                    .unwrap_or(&entry.path())
                                                    .to_string_lossy()
                                                    .to_string();

                                                packages.push(PackageInfo {
                                                    name: name.to_string(),
                                                    path: rel_path,
                                                    dependencies: deps,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Also check common directories if not found via workspaces: apps, packages, libs
    if packages.is_empty() {
        for group in &["packages", "apps", "libs"] {
            let group_dir = root.join(group);
            if group_dir.is_dir() {
                if let Ok(entries) = fs::read_dir(group_dir) {
                    for entry in entries.flatten() {
                        let sub_pkg = entry.path().join("package.json");
                        let sub_py = entry.path().join("pyproject.toml");
                        let sub_cargo = entry.path().join("Cargo.toml");

                        let mut name = entry.file_name().to_string_lossy().to_string();
                        let mut deps = Vec::new();

                        if sub_pkg.exists() {
                            if let Ok(c) = fs::read_to_string(&sub_pkg) {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&c) {
                                    if let Some(n) = v.get("name").and_then(|n| n.as_str()) {
                                        name = n.to_string();
                                    }
                                    for dep_field in &["dependencies", "devDependencies", "peerDependencies"] {
                                        if let Some(obj) = v.get(dep_field).and_then(|d| d.as_object()) {
                                            for k in obj.keys() {
                                                deps.push(k.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let rel_path = entry
                            .path()
                            .strip_prefix(root)
                            .unwrap_or(&entry.path())
                            .to_string_lossy()
                            .to_string();

                        if sub_pkg.exists() || sub_py.exists() || sub_cargo.exists() {
                            packages.push(PackageInfo {
                                name,
                                path: rel_path,
                                dependencies: deps,
                            });
                        }
                    }
                }
            }
        }
    }

    WorkspaceReport {
        total_packages: packages.len(),
        workspace_type: ws_type,
        packages,
    }
}

pub fn get_blast_radius_rs(root_dir: &str, target_file: &str) -> BlastRadiusReport {
    let ws = scan_workspace_rs(root_dir);
    let target_clean = target_file.trim_start_matches("./");

    // Find which package owns this file
    let mut owning_pkg: Option<&PackageInfo> = None;
    for pkg in &ws.packages {
        if target_clean.starts_with(&pkg.path) {
            owning_pkg = Some(pkg);
            break;
        }
    }

    let mut direct_dependents = Vec::new();
    let mut affected_paths = Vec::new();

    if let Some(owner) = owning_pkg {
        affected_paths.push(owner.path.clone());
        let owner_name = &owner.name;

        for candidate in &ws.packages {
            if candidate.path == owner.path {
                continue;
            }
            if candidate.dependencies.contains(owner_name) {
                direct_dependents.push(candidate.name.clone());
                affected_paths.push(candidate.path.clone());
            }
        }
    }

    BlastRadiusReport {
        file_path: target_file.to_string(),
        owning_package: owning_pkg.map(|p| p.name.clone()),
        direct_dependents,
        affected_package_paths: affected_paths,
    }
}
