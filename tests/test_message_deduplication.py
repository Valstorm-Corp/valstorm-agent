"""Unit tests for message deduplication and repeating text collapse."""

import pytest
from core.models import Message, SessionState, collapse_repeating_text
from providers.gemini import GeminiProvider
import server


def test_collapse_repeating_text_exact_multiples():
    sample_paragraph = (
        "### 🎯 Updated & Grounded: `file_name` Added & CLI Sync Compatibility Guaranteed!\n\n"
        "Every serverless Python function record must have its file_name property populated.\n"
    )
    # Test 2x repetition
    text_2x = sample_paragraph * 2
    assert collapse_repeating_text(text_2x) == sample_paragraph

    # Test 4x repetition (like user's snippet)
    text_4x = sample_paragraph * 4
    assert collapse_repeating_text(text_4x) == sample_paragraph

    # Test 8x repetition
    text_8x = sample_paragraph * 8
    assert collapse_repeating_text(text_8x) == sample_paragraph

    # Test 3x repetition
    text_3x = sample_paragraph * 3
    assert collapse_repeating_text(text_3x) == sample_paragraph


def test_collapse_repeating_text_delimiter_separated():
    block = "This is a detailed summary paragraph that was repeated multiple times by the model."
    text_newlines = "\n\n".join([block] * 3)
    assert collapse_repeating_text(text_newlines) == block

    text_single_nl = "\n".join([block] * 4)
    assert collapse_repeating_text(text_single_nl) == block


def test_collapse_repeating_text_no_false_positives():
    # Normal short strings
    assert collapse_repeating_text("Hello world") == "Hello world"
    assert collapse_repeating_text(None) is None
    assert collapse_repeating_text("") == ""

    # Normal article with repeated words or headings
    normal_text = (
        "# Introduction\n\nThis is paragraph one.\n\n"
        "# Details\n\nThis is paragraph two, which has more details.\n\n"
        "# Conclusion\n\nAll tasks finished successfully."
    )
    assert collapse_repeating_text(normal_text) == normal_text


def test_gemini_format_messages_deduplicates_identical_parts():
    provider = GeminiProvider(api_key="fake_key")
    messages = [
        Message(role="user", content="Deploy changes"),
        Message(role="assistant", content="### Status\nDeployment complete and verified."),
        # Duplicate assistant message from cloud hydration
        Message(role="assistant", content="### Status\nDeployment complete and verified."),
    ]
    _, contents = provider._format_messages(messages)

    # Must contain 1 user content and 1 model content
    model_contents = [c for c in contents if c.role == "model"]
    assert len(model_contents) == 1
    # Model content must have exactly 1 text part (the duplicate was stripped!)
    text_parts = [p.text for p in model_contents[0].parts if getattr(p, "text", None)]
    assert len(text_parts) == 1
    assert "Deployment complete" in text_parts[0]


def test_gemini_format_messages_collapses_repeating_history():
    provider = GeminiProvider(api_key="fake_key")
    repeated_content = "### 🎯 All systems nominal and verified across all test runs.\n" * 4
    messages = [
        Message(role="user", content="Check status"),
        Message(role="assistant", content=repeated_content),
    ]
    _, contents = provider._format_messages(messages)
    model_contents = [c for c in contents if c.role == "model"]
    assert len(model_contents) == 1
    text = model_contents[0].parts[0].text
    # Should be collapsed to single copy
    assert text == "### 🎯 All systems nominal and verified across all test runs.\n"


@pytest.mark.asyncio
async def test_hydrate_session_from_cloud_deduplication(monkeypatch):
    session = SessionState(session_id="aich_dedup_test", active_model="gemini-3.6-flash", active_provider="gemini")
    session.add_message(Message(role="system", content="System prompt"))
    session.add_message(Message(id="aicm_local_user_1", role="user", content="Run analysis on tickets"))
    session.add_message(Message(id="aicm_local_asst_1", role="assistant", content="Analysis complete: found 4 tickets."))

    # Cloud has:
    # 1. The same user message with different cloud ID (e.g. created by web UI)
    # 2. The same assistant message with cloud run ID (e.g. aicm_run_...)
    # 3. A genuinely new assistant message (e.g. from /help or support)
    mock_cloud_records = [
        {"id": "aicm_cloud_user_1", "role": "user", "body": "Run analysis on tickets", "created_date": "2026-09-13T10:00:00Z"},
        {"id": "aicm_run_123456_asst", "role": "assistant", "body": "Analysis complete: found 4 tickets.", "created_date": "2026-09-13T10:00:05Z"},
        {"id": "aicm_cloud_new_reply", "role": "assistant", "body": "Support agent: Here is additional guidance for your tickets.", "created_date": "2026-09-13T10:05:00Z"},
    ]

    class MockQueryResponse:
        status_code = 200
        def json(self):
            return {"data": mock_cloud_records}

    async def mock_post(self, url, json=None, headers=None):
        return MockQueryResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)

    saved_sessions = []
    class MockSessionStore:
        def save_session(self, s):
            saved_sessions.append(s)

    await server._hydrate_session_from_cloud_if_needed(
        session=session,
        session_id="aich_dedup_test",
        valstorm_token="test_jwt",
        valstorm_base_url="https://api.valstorm.com/v1",
        session_store=MockSessionStore(),
    )

    # Exactly ONE new message should have been added (the genuinely new support reply)
    # The duplicate user message and duplicate assistant message should both be skipped!
    assert len(session.messages) == 4
    assert session.messages[0].role == "system"
    assert session.messages[1].id == "aicm_local_user_1"
    assert session.messages[2].id == "aicm_local_asst_1"
    assert session.messages[3].id == "aicm_cloud_new_reply"
    assert "Support agent" in session.messages[3].content
