"""Unit tests for Workstream 4D: Native Multimodal Vision Pipeline."""

import base64
import os
from pathlib import Path
import pytest

from core.models import Message
from providers.anthropic import AnthropicProvider
from providers.base import extract_and_resolve_images
from providers.gemini import GeminiProvider
from providers.openai import OpenAIProvider


@pytest.fixture
def sample_image_file(tmp_path):
    """Creates a temporary sample PNG image file."""
    img_path = tmp_path / "test_screenshot.png"
    # Minimal 1x1 valid transparent PNG bytes
    png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    img_path.write_bytes(png_bytes)
    return img_path


def test_extract_image_attachments_explicit_list(sample_image_file):
    """Verify extracting images from Message.images list."""
    msg = Message(role="user", content="Analyze this screenshot", images=[str(sample_image_file)])
    images = extract_and_resolve_images(msg)

    assert len(images) == 1
    raw_bytes, mime = images[0]
    assert mime == "image/png"
    assert len(raw_bytes) > 0


def test_extract_image_attachments_prompt_tag(sample_image_file):
    """Verify extracting image paths from [Attached Image: ...] prompt tags."""
    content = f"Please inspect the layout [Attached Image: {sample_image_file}] and tell me if the button is aligned."
    msg = Message(role="user", content=content)
    images = extract_and_resolve_images(msg)

    assert len(images) == 1
    raw_bytes, mime = images[0]
    assert mime == "image/png"


def test_extract_image_attachments_data_uri():
    """Verify extracting inline base64 data URIs."""
    data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    msg = Message(role="user", content="Here is an image", images=[data_uri])
    images = extract_and_resolve_images(msg)

    assert len(images) == 1
    raw_bytes, mime = images[0]
    assert mime == "image/jpeg"
    assert len(raw_bytes) > 0


def test_gemini_vision_formatting(sample_image_file):
    """Verify Gemini provider transforms images into types.Part inline binary parts."""
    msg = Message(
        role="user",
        content="What is in this mockup?",
        images=[str(sample_image_file)],
    )
    provider = GeminiProvider(api_key="mock_key")
    system_inst, contents = provider._format_messages([msg])

    assert len(contents) == 1
    parts = contents[0].parts
    assert len(parts) == 2  # Text part + Image part
    assert parts[0].text == "What is in this mockup?"
    assert hasattr(parts[1], "inline_data") or hasattr(parts[1], "mime_type") or str(type(parts[1]))


def test_openai_vision_formatting(sample_image_file):
    """Verify OpenAI provider formats images as image_url items in content array."""
    msg = Message(
        role="user",
        content="Analyze UI",
        images=[str(sample_image_file)],
    )
    provider = OpenAIProvider(api_key="mock_key")
    formatted = provider._format_messages([msg])

    assert len(formatted) == 1
    user_msg = formatted[0]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "image_url"
    assert "data:image/png;base64," in user_msg["content"][1]["image_url"]["url"]


def test_anthropic_vision_formatting(sample_image_file):
    """Verify Anthropic provider formats images as image source blocks."""
    msg = Message(
        role="user",
        content="Describe diagram",
        images=[str(sample_image_file)],
    )
    provider = AnthropicProvider(api_key="mock_key")
    sys_prompt, formatted = provider._format_messages([msg])

    assert len(formatted) == 1
    user_msg = formatted[0]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "image"
    assert user_msg["content"][0]["source"]["type"] == "base64"
    assert user_msg["content"][0]["source"]["media_type"] == "image/png"
    assert user_msg["content"][1]["type"] == "text"


@pytest.mark.asyncio
async def test_analyze_image_tool_execution(sample_image_file):
    """Verify analyze_image tool executes cleanly with provider override."""
    from tools.vision_tool import analyze_image, set_vision_provider_override, vision_analyze

    class MockVisionProvider:
        async def generate_response(self, messages, model=None, **kwargs):
            return Message(
                role="assistant",
                content="Mock visual analysis: Observed a clear 1x1 image layout with normal rendering.",
                model=model,
            )

    set_vision_provider_override(MockVisionProvider())
    try:
        res = await analyze_image(path=str(sample_image_file), prompt="Check UI layout")
        assert "Visual Analysis" in res
        assert "test_screenshot.png" in res
        assert "Mock visual analysis" in res

        # Test vision_analyze alias
        res_alias = await vision_analyze(path=str(sample_image_file), prompt="Inspect diagram")
        assert "Visual Analysis" in res_alias
        assert "Mock visual analysis" in res_alias
    finally:
        set_vision_provider_override(None)


@pytest.mark.asyncio
async def test_analyze_image_missing_and_invalid_files(tmp_path):
    """Verify analyze_image handles non-existent and unsupported files gracefully."""
    from tools.vision_tool import analyze_image

    res_missing = await analyze_image(path="/non/existent/image.png")
    assert "Error: Image file not found" in res_missing

    empty_txt = tmp_path / "test.txt"
    empty_txt.write_text("not an image")
    res_unsupported = await analyze_image(path=str(empty_txt))
    assert "Error: Unsupported image extension" in res_unsupported

    res_empty = await analyze_image(path="")
    assert "Error: An image file path must be provided" in res_empty


@pytest.mark.asyncio
async def test_extract_and_resolve_images_vfs(monkeypatch):
    """Verify extract_and_resolve_images resolves cloud://file_... and VFS references."""
    import httpx
    from unittest.mock import MagicMock

    png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

    # Mock resolve_valstorm_auth_context
    monkeypatch.setattr("tools.valstorm_client.resolve_valstorm_auth_context", lambda **kwargs: ("mock_token", "http://localhost:8000", "", ""))

    # Mock httpx.Client response
    class MockResponse:
        def __init__(self, status_code, json_data=None, content=b""):
            self.status_code = status_code
            self._json = json_data or {}
            self.content = content
            self.headers = {"content-type": "image/png"}
        def json(self):
            return self._json

    def mock_get(self, url, headers=None, **kwargs):
        if "/vfs/file/file_sample123" in url:
            return MockResponse(200, json_data={"id": "file_sample123", "name": "card.png", "extension": "png", "url": "http://s3.local/img.png"})
        elif url == "http://s3.local/img.png":
            return MockResponse(200, content=png_bytes)
        return MockResponse(404)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    msg = Message(role="user", content="Here is my card [Attached Image: cloud://file_sample123]")
    images = extract_and_resolve_images(msg)

    assert len(images) == 1
    raw_bytes, mime = images[0]
    assert mime == "image/png"
    assert raw_bytes == png_bytes


@pytest.mark.asyncio
async def test_analyze_image_cloud_vfs_file(monkeypatch):
    """Verify analyze_image tool can analyze images via cloud:// file ID directly."""
    import httpx
    from tools.vision_tool import analyze_image, set_vision_provider_override
    from unittest.mock import AsyncMock, MagicMock

    png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

    monkeypatch.setattr("tools.valstorm_client.resolve_valstorm_auth_context", lambda **kwargs: ("mock_token", "http://localhost:8000", "", ""))

    class MockResponse:
        def __init__(self, status_code, json_data=None, content=b""):
            self.status_code = status_code
            self._json = json_data or {}
            self.content = content
            self.headers = {"content-type": "image/png"}
        def json(self):
            return self._json

    def mock_get(self, url, headers=None, **kwargs):
        if "/vfs/file/file_card999" in url:
            return MockResponse(200, json_data={"id": "file_card999", "name": "card.png", "extension": "png", "url": "http://s3.local/card.png"})
        elif url == "http://s3.local/card.png":
            return MockResponse(200, content=png_bytes)
        return MockResponse(404)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(
        return_value=(Message(role="assistant", content="Business card: Barry W. Butlien, CFP"), None)
    )
    set_vision_provider_override(mock_provider)

    try:
        res = await analyze_image(path="cloud://file_card999", prompt="Read business card")
        assert "Visual Analysis" in res
        assert "Barry W. Butlien, CFP" in res
    finally:
        set_vision_provider_override(None)


def test_extract_and_resolve_images_ignores_vfs_documents_and_octet_stream(monkeypatch):
    """Verify markdown docs and files with application/octet-stream in VFS are ignored by image extractor."""
    import httpx

    monkeypatch.setattr("tools.valstorm_client.resolve_valstorm_auth_context", lambda **kwargs: ("mock_token", "http://localhost:8000", "", ""))

    class MockResponse:
        def __init__(self, status_code, json_data=None):
            self.status_code = status_code
            self._json = json_data or {}
        def json(self):
            return self._json

    def mock_get(self, url, headers=None, **kwargs):
        if "file_summary_octet" in url:
            return MockResponse(200, json_data={
                "id": "file_summary_octet",
                "name": "2026_10_09_summary.md",
                "extension": "md",
                "content_type": "application/octet-stream",
                "text_content": "# Telemetry Summary",
            })
        if "file_doc_markdown" in url:
            return MockResponse(200, json_data={
                "id": "file_doc_markdown",
                "name": "Offer.md",
                "extension": "md",
                "content_type": "text/markdown",
                "text_content": "# Commercial Offer",
            })
        return MockResponse(404)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    prompt = (
        "Here are docs:\n"
        "1. [Offer](cloud://file_doc_markdown)\n"
        "2. [Summary](cloud://file_summary_octet)\n"
        "Please read them."
    )
    msg = Message(role="user", content=prompt)
    images = extract_and_resolve_images(msg)

    # Neither markdown file should be treated as an image
    assert len(images) == 0


def test_extract_and_resolve_images_standalone_local_path(sample_image_file):
    """Verify standalone local file paths on their own line are extracted as images."""
    prompt = f"Here is my issue\n{sample_image_file}\nWhat do you see?"
    msg = Message(role="user", content=prompt)
    images = extract_and_resolve_images(msg)

    assert len(images) == 1
    raw_bytes, mime = images[0]
    assert mime == "image/png"


def test_gemini_vision_rejects_application_octet_stream(sample_image_file, monkeypatch):
    """Verify Gemini provider defensively drops application/octet-stream parts."""
    msg = Message(
        role="user",
        content="Check this",
        images=[str(sample_image_file)],
    )
    # Simulate an image resolved with application/octet-stream
    monkeypatch.setattr("providers.gemini.extract_and_resolve_images", lambda m: [(b"binary_data", "application/octet-stream")])

    provider = GeminiProvider(api_key="mock_key")
    system_inst, contents = provider._format_messages([msg])

    assert len(contents) == 1
    # Only the text part should exist; the application/octet-stream part must be omitted
    parts = contents[0].parts
    assert len(parts) == 1
    assert parts[0].text == "Check this"


def test_extract_and_resolve_images_ignores_local_video_paths(tmp_path):
    """Verify local .mp4/.mov video files pasted in prompts or images list are never Base64-inlined."""
    cam_video = tmp_path / "camera_2026-10-16 09-02-19.mp4"
    screen_video = tmp_path / "screen_2026-10-16 09-02-19.mp4"
    cam_video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 1024)
    screen_video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"1" * 1024)

    prompt = (
        "I have these two movies files. One has the audio, the screen, the other is a camera of me.\n\n"
        f"{cam_video}\n\n"
        f"{screen_video}\n\n"
        "I want you to leverage the video editor to turn this into a piece of content."
    )
    msg = Message(role="user", content=prompt, images=[str(cam_video)])
    images = extract_and_resolve_images(msg)
    assert images == []


def test_resolve_image_source_enforces_max_inline_media_bytes(tmp_path, monkeypatch):
    """Verify oversized local image files exceeding MAX_INLINE_MEDIA_BYTES are not read into memory."""
    import providers.base as base_mod
    from providers.base import resolve_image_source

    monkeypatch.setattr(base_mod, "MAX_INLINE_MEDIA_BYTES", 512)
    oversized_png = tmp_path / "huge.png"
    oversized_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 1024)

    assert resolve_image_source(str(oversized_png)) is None





