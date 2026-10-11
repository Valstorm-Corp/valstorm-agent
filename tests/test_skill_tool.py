"""Unit tests for Skill Discovery and Inspection Tooling (Phase 4 Milestone 1).

Tests:
1. skill_list: Full indexing, category filtering, search queries, pagination.
2. skill_view: Direct slug lookup, category prefix resolution, normalized slugs, suggestions on miss.
3. Subfile and reference document reading with path traversal security.
4. Custom VALSTORM_SKILLS_DIR override resolution.
5. Robust pure-Python frontmatter parsing.
6. Integration into ToolRegistry, create_tier1_tools, and execute_code runtime.
"""

import asyncio
import os
from pathlib import Path
import tempfile
import pytest

from core.tools import ToolRegistry, get_default_registry
from tools import (
    create_tier1_tools,
    execute_code,
    register_tier1_tools,
    skill_list,
    skill_view,
)
from tools.skill_tool import (
    _index_all_skills,
    _parse_frontmatter,
    _resolve_skills_dir,
    create_skill_tools,
    register_skill_tools,
)


@pytest.fixture
def mock_skills_directory(tmp_path):
    """Creates a mock skills repository fixture."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    # Category 1: software-development
    tdd_dir = skills_root / "software-development" / "test-driven-development"
    tdd_dir.mkdir(parents=True)
    (tdd_dir / "SKILL.md").write_text(
        """---
name: test-driven-development
description: "TDD: enforce RED-GREEN-REFACTOR, tests before code."
version: 1.1.0
author: Valstorm
metadata:
  hermes:
    tags: [testing, tdd, quality, red-green-refactor]
---

# Test-Driven Development (TDD)

## Overview
Write the test first. Watch it fail. Write minimal code to pass.
""",
        encoding="utf-8",
    )

    # Add supporting subfile
    refs_dir = tdd_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "pytest_guide.md").write_text("# Pytest Guide\nRun pytest with -v.", encoding="utf-8")

    # Category 2: research
    arxiv_dir = skills_root / "research" / "arxiv"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "SKILL.md").write_text(
        """---
name: arxiv
description: "Search and download papers from arXiv."
version: 1.0.0
author: Valstorm
metadata:
  hermes:
    tags: [research, papers, arxiv, literature]
---

# arXiv Paper Tooling
Query the arXiv API for preprints.
""",
        encoding="utf-8",
    )

    # Category 3: devops
    sdlc_dir = skills_root / "devops" / "sdlc-review"
    sdlc_dir.mkdir(parents=True)
    (sdlc_dir / "SKILL.md").write_text(
        """---
name: sdlc-review
description: "Review SDLC and pull request safety guidelines."
version: 2.0.0
metadata:
  hermes:
    tags: [devops, git, pr, review]
---

# SDLC Review SOP
Inspect diffs before merge.
""",
        encoding="utf-8",
    )

    return skills_root


def test_frontmatter_parser():
    """Verify parsing of frontmatter metadata and body."""
    raw = """---
name: sample-skill
description: "Sample description text"
version: 1.2.3
author: Test Author
metadata:
  hermes:
    tags: [alpha, beta, gamma]
---

# Sample Body Title
Body content goes here.
"""
    meta, body = _parse_frontmatter(raw)
    assert meta["name"] == "sample-skill"
    assert meta["description"] == "Sample description text"
    assert meta["version"] == "1.2.3"
    assert meta["author"] == "Test Author"
    assert meta["metadata"]["hermes"]["tags"] == ["alpha", "beta", "gamma"]
    assert "# Sample Body Title" in body
    assert "Body content goes here." in body


def test_frontmatter_parser_no_frontmatter():
    """Verify fallback behavior when markdown lacks frontmatter."""
    raw = "# Plain Markdown\nNo frontmatter here."
    meta, body = _parse_frontmatter(raw)
    assert meta == {}
    assert body == raw


@pytest.mark.asyncio
async def test_skill_list_all(mock_skills_directory, monkeypatch):
    """Verify skill_list lists all available skills in mock repo."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_list()
    assert "Available Agent Skills" in res
    assert "software-development/test-driven-development" in res
    assert "research/arxiv" in res
    assert "devops/sdlc-review" in res
    assert "TDD: enforce RED-GREEN-REFACTOR" in res


@pytest.mark.asyncio
async def test_skill_list_filter_category(mock_skills_directory, monkeypatch):
    """Verify skill_list filters by category."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_list(category="research")
    assert "research/arxiv" in res
    assert "test-driven-development" not in res
    assert "sdlc-review" not in res


@pytest.mark.asyncio
async def test_skill_list_filter_query(mock_skills_directory, monkeypatch):
    """Verify skill_list searches by keyword across slug, description, and tags."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    # Match by tag
    res_tag = await skill_list(query="tdd")
    assert "test-driven-development" in res_tag
    assert "arxiv" not in res_tag

    # Match by description word
    res_desc = await skill_list(query="preprints")
    assert "research/arxiv" in res_desc
    assert "test-driven-development" not in res_desc


@pytest.mark.asyncio
async def test_skill_list_no_matches(mock_skills_directory, monkeypatch):
    """Verify skill_list returns descriptive message when no match is found."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_list(query="nonexistent_xyz_query")
    assert "No skills found" in res
    assert "nonexistent_xyz_query" in res


@pytest.mark.asyncio
async def test_skill_view_exact_slug(mock_skills_directory, monkeypatch):
    """Verify skill_view loads and renders exact skill slug."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_view("test-driven-development")
    assert "=== SKILL: software-development/test-driven-development" in res
    assert "Write the test first. Watch it fail." in res


@pytest.mark.asyncio
async def test_skill_view_with_category_prefix(mock_skills_directory, monkeypatch):
    """Verify skill_view handles 'category/slug' format."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_view("research/arxiv")
    assert "=== SKILL: research/arxiv" in res
    assert "Query the arXiv API for preprints." in res


@pytest.mark.asyncio
async def test_skill_view_normalized_case_and_underscores(mock_skills_directory, monkeypatch):
    """Verify skill_view handles underscores and case variations."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_view("TEST_DRIVEN_DEVELOPMENT")
    assert "=== SKILL: software-development/test-driven-development" in res
    assert "Write the test first." in res


@pytest.mark.asyncio
async def test_skill_view_not_found_with_suggestions(mock_skills_directory, monkeypatch):
    """Verify skill_view suggests closest matches when a skill is not found."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_view("test-driven")
    # Substring match will directly resolve or suggest
    assert "test-driven-development" in res

    res_miss = await skill_view("unknown-nonexistent-skill")
    assert "Error: Skill 'unknown-nonexistent-skill' not found" in res_miss


@pytest.mark.asyncio
async def test_skill_view_subfile(mock_skills_directory, monkeypatch):
    """Verify skill_view can read reference subfiles in the skill directory."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_view("test-driven-development", file_path="references/pytest_guide.md")
    assert "# Pytest Guide" in res
    assert "Run pytest with -v." in res


@pytest.mark.asyncio
async def test_skill_view_path_traversal_blocked(mock_skills_directory, monkeypatch):
    """Verify skill_view blocks directory traversal attempts via file_path."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    res = await skill_view("test-driven-development", file_path="../../secrets.txt")
    assert "Error: Access denied" in res or "traverses outside" in res


def test_skill_tools_registry_registration():
    """Verify skill_view and skill_list are registered in ToolRegistry and create_tier1_tools."""
    registry = get_default_registry()
    register_skill_tools(registry)

    assert "skill_view" in registry.list_tools()
    assert "skill_list" in registry.list_tools()

    schemas = {s["name"]: s for s in registry.get_schemas()}
    assert "skill_view" in schemas
    assert "skill_list" in schemas

    tier1 = create_tier1_tools()
    tier1_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in tier1]
    assert "skill_view" in tier1_names
    assert "skill_list" in tier1_names


@pytest.mark.asyncio
async def test_execute_code_integration_with_skills(mock_skills_directory, monkeypatch):
    """Verify that execute_code scripts can call skill_list and skill_view synchronously."""
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(mock_skills_directory))

    script = """
skills_text = skill_list()
print("LIST_LEN:", len(skills_text))
view_text = skill_view("test-driven-development")
print("HAS_TDD:", "RED-GREEN-REFACTOR" in view_text)
"""
    result = await execute_code(code=script)
    assert "LIST_LEN:" in result
    assert "HAS_TDD: True" in result
