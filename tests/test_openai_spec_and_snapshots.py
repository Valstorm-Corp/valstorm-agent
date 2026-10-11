"""Tests for Universal OpenAI-Compatible Spec Provider Instantiation and Git Snapshots."""

from pathlib import Path
import pytest

from core.keystore import KeyStore
from core.snapshot import GitSnapshotManager, get_snapshot_manager
from providers import (
    OPENAI_COMPATIBLE_PRESETS,
    OpenAIProvider,
    resolve_provider_instance,
)


def test_openai_compatible_presets():
    assert "deepseek" in OPENAI_COMPATIBLE_PRESETS
    assert "groq" in OPENAI_COMPATIBLE_PRESETS
    assert "moonshot" in OPENAI_COMPATIBLE_PRESETS
    assert "ollama" in OPENAI_COMPATIBLE_PRESETS
    assert "vllm" in OPENAI_COMPATIBLE_PRESETS

    deepseek_preset = OPENAI_COMPATIBLE_PRESETS["deepseek"]
    assert deepseek_preset["base_url"] == "https://api.deepseek.com/v1"
    assert deepseek_preset["default_model"] == "deepseek-chat"

    groq_preset = OPENAI_COMPATIBLE_PRESETS["groq"]
    assert groq_preset["base_url"] == "https://api.groq.com/openai/v1"

    ollama_preset = OPENAI_COMPATIBLE_PRESETS["ollama"]
    assert ollama_preset["base_url"] == "http://localhost:11434/v1"
    assert ollama_preset["default_key"] == "ollama"


def test_resolve_provider_instance_deepseek():
    provider, model, prov_name = resolve_provider_instance("deepseek", api_key="sk-test-deepseek")
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert model == "deepseek-chat"
    assert prov_name == "deepseek"


def test_resolve_provider_instance_ollama():
    provider, model, prov_name = resolve_provider_instance("ollama")
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.api_key == "ollama"
    assert model == "qwen2.5-coder:32b"
    assert prov_name == "ollama"


def test_keystore_provider_mapping():
    keystore = KeyStore()
    candidates = keystore._get_candidate_keys_for_provider("deepseek")
    assert "DEEPSEEK_API_KEY" in candidates

    groq_candidates = keystore._get_candidate_keys_for_provider("groq")
    assert "GROQ_API_KEY" in groq_candidates


@pytest.mark.asyncio
async def test_git_snapshot_creation_and_revert(tmp_path: Path):
    manager = GitSnapshotManager(workdir=tmp_path)
    # When not a git repo, gracefully returns None
    snap = await manager.create_snapshot("test_sess", turn_index=1)
    assert snap is None

    success, msg = await manager.revert_turn("test_sess", turn_index=1)
    assert success is False
    assert "not a Git repository" in msg
