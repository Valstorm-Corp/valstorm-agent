"""Tests for Persistent Declarative Memory and Session Search (Storage Engine)."""

import json
from pathlib import Path
import pytest

from core.memory import MemoryStore
from core.tools import ToolRegistry
from tools.memory_tool import register_memory_tools


def test_memory_store_local_crud(tmp_path: Path):
    mem_file = tmp_path / "test_memories.json"
    store = MemoryStore(file_path=mem_file)

    # 1. Add user facts
    store.add_fact("user", "Prefers TypeScript over plain JavaScript.")
    store.add_fact("user", "Works on macOS Apple Silicon.")

    # 2. Add memory / workspace facts
    store.add_fact("memory", "Monorepo uses yarn v1 and python uv.")

    # 3. Verify state
    facts = store.get_facts()
    assert len(facts["user"]) == 2
    assert len(facts["memory"]) == 1
    assert "Prefers TypeScript" in facts["user"][0]

    # 4. Remove fact
    removed = store.remove_fact("user", "Apple Silicon")
    assert removed is True
    facts_after = store.get_facts()
    assert len(facts_after["user"]) == 1

    # 5. Format for prompt
    prompt_snippet = store.format_for_system_prompt()
    assert "Long-Term Memory" in prompt_snippet or "Persistent Declarative Memory" in prompt_snippet
    assert "Prefers TypeScript" in prompt_snippet
    assert "Monorepo uses yarn v1" in prompt_snippet


def test_memory_tools_execution(tmp_path):
    mem_file = tmp_path / "test_memories.json"
    store = MemoryStore(file_path=mem_file)
    registry = ToolRegistry()
    register_memory_tools(registry, memory_store=store)

    assert "memory_manage" in registry.list_tools()
    assert "session_search" in registry.list_tools()

    # Execute memory_manage add
    add_res = registry.execute("memory_manage", args={"action": "add", "target": "user", "content": "Jared is the lead engineer."})
    assert "Successfully added" in add_res.output or "Saved fact to tenant memory" in add_res.output

    # Execute memory_manage list
    list_res = registry.execute("memory_manage", args={"action": "list", "target": "user"})
    assert "Jared is the lead engineer." in list_res.output

    # Execute memory_manage remove
    remove_res = registry.execute("memory_manage", args={"action": "remove", "target": "user", "old_text": "lead engineer"})
    assert "Successfully removed" in remove_res.output or "Removed fact" in remove_res.output
