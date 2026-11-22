"""Regression tests for ChainedFallbackProvider.chat_complete and ValstormApiClient /v1/v1 normalization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory_extractor import extract_memories_from_turn
from core.models import Message, UsageMetadata
from core.skill_curator import curate_and_commit_turn_trajectory
from providers.base import BaseProvider
from providers.fallback import ChainedFallbackProvider, FallbackTier
from tools.valstorm_client import ValstormApiClient


class _DummyProvider(BaseProvider):
    def __init__(self, reply_text: str):
        super().__init__(default_model="dummy-worker")
        self._reply_text = reply_text

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageMetadata]:
        return Message(role="assistant", content=self._reply_text), UsageMetadata(input_tokens=10, output_tokens=5)


@pytest.mark.asyncio
async def test_chained_fallback_provider_supports_chat_complete():
    tier = FallbackTier(
        provider=_DummyProvider('{"action": "noop"}'),
        model="dummy-worker",
        provider_name="dummy",
        label="Dummy Worker",
    )
    chain = ChainedFallbackProvider(tiers=[tier])

    msg = await chain.chat_complete(
        messages=[Message(role="user", content="hello")],
        model="dummy-worker",
        temperature=0.0,
    )
    assert isinstance(msg, Message)
    assert msg.content == '{"action": "noop"}'


@pytest.mark.asyncio
async def test_skill_curator_and_memory_extractor_work_with_chained_fallback_provider():
    tier_skill = FallbackTier(
        provider=_DummyProvider(json.dumps({"action": "noop"})),
        model="dummy-worker",
        provider_name="dummy",
        label="Dummy Skill Worker",
    )
    chain_skill = ChainedFallbackProvider(tiers=[tier_skill])

    curator_res = await curate_and_commit_turn_trajectory(
        user_input="Fix the failing deployment pipeline and verify tests.",
        assistant_output="Fixed and verified all tests pass.",
        session=None,
        provider=chain_skill,
        model="dummy-worker",
        force=True,
    )
    assert curator_res["status"] == "noop"

    tier_mem = FallbackTier(
        provider=_DummyProvider(json.dumps([{"category": "user", "fact": "Prefers pytest over unittest"}])),
        model="dummy-worker",
        provider_name="dummy",
        label="Dummy Memory Worker",
    )
    chain_mem = ChainedFallbackProvider(tiers=[tier_mem])
    facts = await extract_memories_from_turn(
        provider=chain_mem,
        model="dummy-worker",
        user_input="I always prefer pytest over unittest for Python services.",
        assistant_output="Noted, I will use pytest.",
        allow_s1_fastpath=False,
    )
    assert len(facts) == 1
    assert facts[0]["fact"] == "Prefers pytest over unittest"


@pytest.mark.asyncio
async def test_valstorm_api_client_prevents_double_v1_prefix():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"api_name": "ai_agent"})

    async with httpx.AsyncClient(
        base_url="https://api.valstorm.com/v1/",
        transport=httpx.MockTransport(handler),
    ) as mock_http:
        client = ValstormApiClient(
            token="pat_test_123",
            base_url="https://api.valstorm.com/v1",
            client=mock_http,
        )
        await client._request_with_retry("GET", "/v1/schema/ai_agent")
        await client._request_with_retry("GET", "v1/schema/ai_agent")
        await client._request_with_retry("GET", "/schema/ai_agent")

    assert seen_paths == [
        "/v1/schema/ai_agent",
        "/v1/schema/ai_agent",
        "/v1/schema/ai_agent",
    ]
