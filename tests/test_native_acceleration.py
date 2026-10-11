"""Unit tests for the Rust Native Acceleration Core (vsagent_native)."""

import json
from pathlib import Path
import pytest

from core.native_bridge import (
    is_native_available,
    native_find_symbols,
    native_generate_outline,
    native_get_blast_radius,
    native_patch_content,
    native_sanitize_text,
    native_scan_workspace,
    native_search_files,
)
from core.sanitizer import sanitize_text
from tools.developer_tools import (
    code_outline,
    find_symbols,
    package_blast_radius,
    patch_file,
    search_files,
)


def test_native_module_loaded():
    """Verifies that vsagent_native is compiled and accessible."""
    assert is_native_available() is True


def test_native_search_files_content(tmp_path: Path):
    """Verifies in-process native regex content search."""
    file1 = tmp_path / "hello.py"
    file1.write_text("def my_special_function():\n    return 42\n", encoding="utf-8")

    file2 = tmp_path / "test.ts"
    file2.write_text("export const my_special_constant = 'test';\n", encoding="utf-8")

    results = native_search_files(
        pattern="my_special",
        target="content",
        path=str(tmp_path),
    )
    assert results is not None
    assert len(results) == 2
    assert any("hello.py:1:" in r for r in results)
    assert any("test.ts:1:" in r for r in results)


def test_native_search_files_target_files(tmp_path: Path):
    """Verifies in-process native file name search."""
    (tmp_path / "service.py").write_text("# service", encoding="utf-8")
    (tmp_path / "service_test.py").write_text("# test", encoding="utf-8")
    (tmp_path / "other.txt").write_text("# other", encoding="utf-8")

    results = native_search_files(
        pattern="service",
        target="files",
        path=str(tmp_path),
    )
    assert results is not None
    assert len(results) == 2
    assert "service.py" in results
    assert "service_test.py" in results


def test_native_patch_content_exact():
    """Verifies native exact patching and diff generation."""
    content = "line 1\nline 2\nline 3\n"
    res = native_patch_content(content, "line 2", "line TWO")
    assert res is not None
    patched, diff = res
    assert "line TWO" in patched
    assert "-line 2" in diff
    assert "+line TWO" in diff


def test_native_patch_content_fuzzy_indentation():
    """Verifies that native patcher handles slight indentation and whitespace differences gracefully."""
    content = "    def run():\n        val = 10\n        return val\n"
    # Old string has different indentation / trailing spaces
    old_str = "def run():\n    val = 10"
    new_str = "def run():\n    val = 99"
    res = native_patch_content(content, old_str, new_str)
    assert res is not None
    patched, diff = res
    assert "val = 99" in patched


def test_native_generate_outline_python():
    """Verifies outline generation for Python."""
    code = '''import os
from typing import List

class DataProcessor:
    """Processes datasets."""
    def __init__(self, name: str):
        self.name = name
        self.items = []

    async def run(self, count: int) -> bool:
        """Runs the processor."""
        for i in range(count):
            pass
        return True

DEFAULT_TIMEOUT = 30
'''
    outline = native_generate_outline("processor.py", code)
    assert outline is not None
    assert "class DataProcessor:" in outline
    assert "def __init__(self, name: str): ..." in outline
    assert "async def run(self, count: int) -> bool: ..." in outline
    assert "for i in range" not in outline  # Implementation body stripped


def test_native_generate_outline_typescript():
    """Verifies outline generation for TypeScript."""
    code = '''import React from 'react';

export interface UserProfileProps {
    userId: string;
    isActive: boolean;
}

export const UserProfile: React.FC<UserProfileProps> = (props) => {
    const [data, setData] = useState(null);
    return <div>User</div>;
};

export function fetchUserData(id: string): Promise<any> {
    return fetch(`/api/user/${id}`);
}
'''
    outline = native_generate_outline("UserProfile.tsx", code)
    assert outline is not None
    assert "export interface UserProfileProps" in outline
    assert "export const UserProfile" in outline
    assert "export function fetchUserData" in outline
    assert "useState(null)" not in outline  # Implementation stripped


def test_native_find_symbols(tmp_path: Path):
    """Verifies fast symbol search."""
    (tmp_path / "auth.py").write_text("class Authenticator:\n    pass\n\ndef verify_token():\n    pass\n", encoding="utf-8")
    (tmp_path / "auth.ts").write_text("export interface AuthConfig { key: string; }\nexport function login() {}\n", encoding="utf-8")

    matches = native_find_symbols(str(tmp_path), query="auth")
    assert matches is not None
    names = [m["name"] for m in matches]
    assert "Authenticator" in names
    assert "AuthConfig" in names


def test_native_scan_workspace_and_blast_radius(tmp_path: Path):
    """Verifies monorepo topology detection and blast radius."""
    # Create fake monorepo structure
    pkg_json = {
        "workspaces": ["packages/*", "apps/*"]
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg_json), encoding="utf-8")

    pkg_core = tmp_path / "packages" / "core"
    pkg_core.mkdir(parents=True)
    (pkg_core / "package.json").write_text(json.dumps({"name": "@test/core"}), encoding="utf-8")
    (pkg_core / "index.ts").write_text("export const version = '1.0';", encoding="utf-8")

    app_web = tmp_path / "apps" / "web"
    app_web.mkdir(parents=True)
    (app_web / "package.json").write_text(
        json.dumps({
            "name": "@test/web",
            "dependencies": {"@test/core": "*"}
        }),
        encoding="utf-8"
    )

    ws = native_scan_workspace(str(tmp_path))
    assert ws is not None
    assert ws["total_packages"] == 2

    # Test blast radius of packages/core/index.ts
    blast = native_get_blast_radius(str(tmp_path), "packages/core/index.ts")
    assert blast is not None
    assert blast["owning_package"] == "@test/core"
    assert "@test/web" in blast["direct_dependents"]
    assert "apps/web" in blast["affected_package_paths"]


def test_native_sanitize_text():
    """Verifies fast Rust regex redaction for secrets."""
    raw = "Here is my secret sk-proj-12345678901234567890 and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz"
    sanitized = native_sanitize_text(raw)
    assert sanitized is not None
    assert "sk-proj-12345678901234567890" not in sanitized
    assert "<REDACTED_OPENAI_KEY>" in sanitized
    assert "<REDACTED_BEARER_TOKEN>" in sanitized


@pytest.mark.asyncio
async def test_developer_tools_integration(tmp_path: Path):
    """Verifies developer tools using native backend."""
    test_file = tmp_path / "App.tsx"
    test_file.write_text(
        "export interface AppProps { title: string; }\nexport function App() { return <h1>App</h1>; }\n",
        encoding="utf-8"
    )

    # 1. code_outline tool
    outline_res = await code_outline(str(test_file))
    assert "export interface AppProps" in outline_res
    assert "export function App" in outline_res

    # 2. find_symbols tool
    sym_res = await find_symbols(query="App", path=str(tmp_path))
    assert "AppProps" in sym_res
    assert "App" in sym_res

    # 3. search_files tool (content)
    search_res = await search_files(pattern="export interface", path=str(tmp_path))
    assert "Found 1 matches" in search_res or "AppProps" in search_res
