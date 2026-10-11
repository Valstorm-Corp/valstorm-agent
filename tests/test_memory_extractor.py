"""Tests for Smart Context & Memory Extraction Engine."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from core.memory import MemoryStore, _token_similarity
from core.memory_extractor import (
    is_memory_candidate,
    extract_memories_from_turn,
    extract_and_commit_turn_memory,
)
from core.models import Message


def test_token_similarity():
    s1 = "Prefers 2-space indentation in TypeScript"
    s2 = "Prefers 2 space indentation for TypeScript code"
    sim = _token_similarity(s1, s2)
    assert sim >= 0.7

    s3 = "Database port is 5432"
    assert _token_similarity(s1, s3) < 0.2


def test_is_memory_candidate_heuristics():
    # Negative cases (transient tasks, commands, standard queries) -> 0 tokens spent
    assert not is_memory_candidate("run pytest tests/")
    assert not is_memory_candidate("fix the bug in AuthModal.tsx")
    assert not is_memory_candidate("git status")
    assert not is_memory_candidate("show me the last 5 leads")
    assert not is_memory_candidate("/memorize Always use pnpm")  # Handled directly by /memorize command
    assert not is_memory_candidate("")
    assert not is_memory_candidate(None)

    # Positive cases (preferences, constraints, changes, roles) -> Should trigger extraction
    assert is_memory_candidate("I prefer 2-space indentation for all React components")
    assert is_memory_candidate("Always use uv instead of poetry in this monorepo")
    assert is_memory_candidate("Never push directly to the main branch")
    assert is_memory_candidate("Actually, we switched from Twilio to AWS SNS for SMS")
    assert is_memory_candidate("My role is Chief Technology Officer")
    assert is_memory_candidate("We use PostgreSQL for all microservices")
    assert is_memory_candidate("Our endpoint is https://api.valstorm.com/v2")


def test_memory_store_reconcile_fact(tmp_path):
    mem_file = tmp_path / "test_memories.json"
    store = MemoryStore(file_path=mem_file)

    # 1. Add initial fact
    res1 = store.reconcile_fact(
        target="memory",
        content="We use Twilio for SMS communications",
    )
    assert res1["status"] == "success"
    assert res1["action"] == "added"
    assert "We use Twilio for SMS communications" in store.get_facts("memory")["memory"]

    # 2. Exact duplicate check
    res_dup = store.reconcile_fact(
        target="memory",
        content="We use Twilio for SMS communications",
    )
    assert res_dup["status"] == "noop"
    assert res_dup["action"] == "exact_duplicate"
    assert len(store.get_facts("memory")["memory"]) == 1

    # 3. Near-duplicate reconciliation (fuzzy replacement)
    res_fuzzy = store.reconcile_fact(
        target="memory",
        content="We use Twilio for all SMS communications",
    )
    assert res_fuzzy["status"] == "success"
    assert res_fuzzy["action"] == "superseded"
    facts = store.get_facts("memory")["memory"]
    assert len(facts) == 1
    assert "We use Twilio for all SMS communications" in facts

    # 4. Explicit supersession / conflict resolution
    res_super = store.reconcile_fact(
        target="memory",
        content="Transactional SMS provider is AWS SNS (Twilio deprecated)",
        supersedes="Twilio",
    )
    assert res_super["status"] == "success"
    assert res_super["action"] == "superseded"
    facts = store.get_facts("memory")["memory"]
    assert len(facts) == 1
    assert "Transactional SMS provider is AWS SNS (Twilio deprecated)" in facts
    assert not any("Twilio for all SMS" in f for f in facts)


@pytest.mark.asyncio
async def test_extract_memories_from_turn_mock(tmp_path):
    mock_provider = MagicMock()
    mock_provider.chat_complete = AsyncMock(
        return_value=Message(
            role="assistant",
            content='[{"category": "user", "fact": "User prefers concise diffs with strict typing", "supersedes": null}]',
        )
    )

    facts = await extract_memories_from_turn(
        user_input="Make sure you always give me concise diffs with strict typing",
        assistant_output="Got it, I will provide concise typed diffs.",
        provider=mock_provider,
    )

    assert len(facts) == 1
    assert facts[0]["category"] == "user"
    assert "concise diffs" in facts[0]["fact"]


@pytest.mark.asyncio
async def test_extract_and_commit_turn_memory(tmp_path):
    mem_file = tmp_path / "test_memories.json"
    store = MemoryStore(file_path=mem_file)

    mock_provider = MagicMock()
    mock_provider.chat_complete = AsyncMock(
        return_value=Message(
            role="assistant",
            content='```json\n[{"category": "user", "fact": "Prefers uv for Python dependencies", "supersedes": "poetry"}]\n```',
        )
    )

    store.add_fact("user", "Prefers poetry for Python dependency management")

    results = await extract_and_commit_turn_memory(
        user_input="Actually, we switched and now I prefer uv for Python dependencies",
        assistant_output="Understood, I will use uv.",
        memory_store=store,
        provider=mock_provider,
    )

    assert len(results) == 1
    assert results[0]["status"] == "success"
    user_facts = store.get_facts("user")["user"]
    assert len(user_facts) == 1
    assert "Prefers uv for Python dependencies" in user_facts
    assert "poetry" not in user_facts[0]
