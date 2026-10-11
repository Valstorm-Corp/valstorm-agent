"""Live API integration tests for Valstorm Platform Tools."""

import time
import pytest

from core.keystore import KeyStore
from core.models import SessionState
from core.react import ReActEngine
from core.tools import ToolRegistry
from providers.gemini import GeminiProvider
from tools.valstorm_client import ValstormApiClient, resolve_valstorm_credentials
from tools.valstorm_tools import register_valstorm_tools
from tests.conftest import record_live_telemetry


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_valstorm_sql_query_with_agent():
    """Live test verifying Agent executes real valstorm_sql_query against active Valstorm API."""
    valstorm_token, base_url = resolve_valstorm_credentials()
    if not valstorm_token:
        pytest.skip("No Valstorm token found (VALSTORM_API_TOKEN or ~/.valstorm/credentials.json)")

    gemini_key = KeyStore.resolve_key("gemini")
    if not gemini_key:
        pytest.skip("No GEMINI_API_KEY found for agent provider")

    valstorm_client = ValstormApiClient(token=valstorm_token, base_url=base_url)
    registry = ToolRegistry()
    register_valstorm_tools(registry=registry, client=valstorm_client)

    model = "gemini-flash-lite-latest"
    provider = GeminiProvider(api_key=gemini_key)
    engine = ReActEngine(provider=provider, tools=registry)
    session = SessionState(active_model=model, active_provider="gemini")

    tools_called = []
    def on_step(event_type: str, payload):
        if event_type == "tool_call":
            name = getattr(payload, "name", str(payload))
            tools_called.append(name)

    start_t = time.time()
    prompt = "Use valstorm_sql_query to select the id and name of up to 2 users from the 'user' table."
    response = await engine.run_turn(
        session=session,
        user_input=prompt,
        model=model,
        step_callback=on_step,
    )
    duration = time.time() - start_t

    assert "valstorm_sql_query" in tools_called
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0

    record_live_telemetry(
        test_name="Live Valstorm SQL Tool",
        provider="Gemini",
        model=model,
        tools_used=tools_called,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        duration_sec=duration,
    )

    await valstorm_client.close()
