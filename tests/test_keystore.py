"""Unit tests for models, session token aggregation, and KeyStore resolution hierarchy."""

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import pytest

from core.models import Message, SessionState, ToolCall, ToolResult, UsageMetadata
from core.keystore import KeyStore


class TestModels:
    """Tests for Pydantic data models and serialization."""

    def test_usage_metadata_defaults_and_serialization(self):
        usage = UsageMetadata(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cached_tokens=25,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens == 25

        dump = usage.model_dump()
        assert dump["prompt_tokens"] == 100
        assert dump["cached_tokens"] == 25

        # Roundtrip JSON
        json_str = usage.model_dump_json()
        loaded = UsageMetadata.model_validate_json(json_str)
        assert loaded == usage

    def test_tool_call_and_result(self):
        call = ToolCall(name="calculator", arguments={"expr": "2 + 2"})
        assert call.id is not None
        assert len(call.id) > 0
        assert call.name == "calculator"
        assert call.arguments == {"expr": "2 + 2"}

        res = ToolResult(call_id=call.id, name="calculator", output="4", is_error=False)
        assert res.call_id == call.id
        assert res.output == "4"
        assert not res.is_error

        # ToolResult serialization
        res_dump = res.model_dump()
        assert res_dump["call_id"] == call.id
        assert res_dump["is_error"] is False

    def test_message_creation_and_defaults(self):
        msg = Message(role="user", content="Hello world")
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert isinstance(msg.created_date, datetime)
        assert isinstance(msg.modified_date, datetime)
        assert msg.tool_calls is None
        assert msg.usage is None

        # Message with tool calls and usage
        usage = UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        tool_call = ToolCall(name="lookup", arguments={"query": "test"})
        assistant_msg = Message(
            role="assistant",
            content="Looking it up",
            model="gemini-2.5-flash",
            provider="gemini",
            tool_calls=[tool_call],
            usage=usage,
        )
        assert assistant_msg.model == "gemini-2.5-flash"
        assert assistant_msg.provider == "gemini"
        assert assistant_msg.tool_calls is not None
        assert assistant_msg.tool_calls[0].name == "lookup"
        assert assistant_msg.usage is not None
        assert assistant_msg.usage.total_tokens == 15

        # Serialization roundtrip
        json_repr = assistant_msg.model_dump_json()
        reconstructed = Message.model_validate_json(json_repr)
        assert reconstructed.id == assistant_msg.id
        assert reconstructed.usage is not None
        assert reconstructed.usage.prompt_tokens == 10

    def test_session_state_and_token_aggregation(self):
        session = SessionState(
            active_model="gemini-2.5-flash",
            active_provider="gemini",
        )
        assert session.session_id is not None
        assert len(session.messages) == 0
        assert session.total_prompt_tokens == 0
        assert session.total_completion_tokens == 0
        assert session.total_tokens == 0

        # Add message without usage
        user_msg = Message(role="user", content="What is the capital of France?")
        session.add_message(user_msg)
        assert len(session.messages) == 1
        assert session.total_tokens == 0

        # Add message with usage
        assistant_msg1 = Message(
            role="assistant",
            content="Paris",
            model="gemini-2.5-flash",
            provider="gemini",
            usage=UsageMetadata(prompt_tokens=12, completion_tokens=4, total_tokens=16),
        )
        session.add_message(assistant_msg1)
        assert len(session.messages) == 2
        assert session.total_prompt_tokens == 12
        assert session.total_completion_tokens == 4
        assert session.total_tokens == 16

        # Add another message with usage
        assistant_msg2 = Message(
            role="assistant",
            content="Anything else?",
            model="gemini-2.5-flash",
            provider="gemini",
            usage=UsageMetadata(prompt_tokens=20, completion_tokens=6, total_tokens=26),
        )
        session.add_message(assistant_msg2)
        assert len(session.messages) == 3
        assert session.total_prompt_tokens == 32
        assert session.total_completion_tokens == 10
        assert session.total_tokens == 42


class TestKeyStore:
    """Tests for KeyStore API key resolution hierarchy."""

    def test_override_key_precedence(self, tmp_path, monkeypatch):
        # Even if env var, keys.json, and .env.ai.keys exist, override_key must take top priority
        monkeypatch.setenv("GEMINI_API_KEY", "env_gemini_key")

        config_file = tmp_path / "keys.json"
        config_file.write_text(json.dumps({"gemini": "config_gemini_key"}))

        env_file = tmp_path / ".env.ai.keys"
        env_file.write_text("GEMINI_API_KEY=env_file_gemini_key\n")

        keystore = KeyStore(config_path=config_file, env_file_path=env_file)
        resolved = keystore.get_api_key("gemini", override_key="explicit_override_key")
        assert resolved == "explicit_override_key"

    def test_env_var_precedence_over_config_and_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env_openai_key")

        config_file = tmp_path / "keys.json"
        config_file.write_text(json.dumps({"openai": "config_openai_key"}))

        env_file = tmp_path / ".env.ai.keys"
        env_file.write_text("OPENAI_API_KEY=env_file_openai_key\n")

        keystore = KeyStore(config_path=config_file, env_file_path=env_file)
        resolved = keystore.get_api_key("openai")
        assert resolved == "env_openai_key"

    def test_config_json_precedence_over_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

        config_file = tmp_path / "keys.json"
        config_file.write_text(json.dumps({"anthropic": "config_anthropic_key"}))

        env_file = tmp_path / ".env.ai.keys"
        env_file.write_text("ANTHROPIC_API_KEY=env_file_anthropic_key\n")

        keystore = KeyStore(config_path=config_file, env_file_path=env_file)
        resolved = keystore.get_api_key("anthropic")
        assert resolved == "config_anthropic_key"

    def test_dotenv_resolution_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        config_file = tmp_path / "non_existent_keys.json"
        env_file = tmp_path / ".env.ai.keys"
        env_file.write_text("GEMINI_API_KEY=dotenv_gemini_key\n")

        keystore = KeyStore(config_path=config_file, env_file_path=env_file)
        resolved = keystore.get_api_key("gemini")
        assert resolved == "dotenv_gemini_key"

    def test_provider_alias_and_casing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google_env_key")

        keystore = KeyStore(
            config_path=tmp_path / "keys.json",
            env_file_path=tmp_path / ".env.ai.keys",
        )
        assert keystore.get_api_key("gemini") == "google_env_key"
        assert keystore.get_api_key("google") == "google_env_key"

    def test_nonexistent_key_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("UNKNOWN_API_KEY", raising=False)
        keystore = KeyStore(
            config_path=tmp_path / "keys.json",
            env_file_path=tmp_path / ".env.ai.keys",
        )
        assert keystore.get_api_key("unknown_provider") is None

    def test_inline_comment_stripping(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        env_file = tmp_path / ".env.ai.keys"
        env_file.write_text("OPENAI_API_KEY=sk-test-token-12345 # inline comment here\n")

        keystore = KeyStore(
            config_path=tmp_path / "keys.json",
            env_file_path=env_file,
        )
        assert keystore.get_api_key("openai") == "sk-test-token-12345"

    def test_convenience_class_method(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config_file = tmp_path / "keys.json"
        config_file.write_text(json.dumps({"OPENAI_API_KEY": "classmethod_openai_key"}))

        resolved = KeyStore.resolve_key(
            provider="openai",
            config_path=config_file,
            env_file_path=tmp_path / ".env.ai.keys",
        )
        assert resolved == "classmethod_openai_key"
