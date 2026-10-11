"""Base provider abstract interface and multimodal vision helpers for agent runtime."""

from abc import ABC, abstractmethod
import base64
import mimetypes
import os
from pathlib import Path
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from core.models import Message, StreamEvent, StreamEventType, UsageMetadata
from core.retry import (
    compute_backoff_delay,
    execute_stream_with_retry,
    execute_with_retry,
    is_retryable_error,
)

IMAGE_EXT_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".heic": "image/heic",
    ".pdf": "application/pdf",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".3gp": "video/3gpp",
    ".m4v": "video/x-m4v",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


VALID_MULTIMODAL_MIME_TYPES = set(IMAGE_EXT_MIME_MAP.values())


def is_valid_multimodal_mime(mime: Optional[str]) -> bool:
    """Returns True if the MIME type is a valid multimodal media type (image, audio, video, pdf)."""
    if not mime or not isinstance(mime, str):
        return False
    norm = mime.strip().lower()
    if norm in ("application/octet-stream", "text/plain", "text/markdown"):
        return False
    return norm in VALID_MULTIMODAL_MIME_TYPES or norm.startswith(("image/", "video/", "audio/"))


def fetch_vfs_image(file_id: str) -> Optional[Tuple[bytes, str]]:
    """Fetches image bytes and MIME type from Valstorm VFS via REST API.

    Only returns binary content for valid multimodal media types (images, pdf, audio, video).
    Non-multimodal documents (markdown, text, code, json) or files with application/octet-stream
    are ignored so they do not break LLM request schemas.
    """
    try:
        from tools.valstorm_client import resolve_valstorm_auth_context
        import httpx

        clean_id = file_id.replace("cloud://", "").strip()
        if not clean_id:
            return None

        token, base_url, _, _ = resolve_valstorm_auth_context()
        if not base_url:
            return None

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        endpoint = f"{base_url.rstrip('/')}/vfs/file/{clean_id}"
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(endpoint, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json()

            ext = str(data.get("extension") or "").lower().strip()
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            content_type = str(data.get("content_type") or "").strip().lower()

            # Disallow non-multimodal extensions (e.g. .md, .txt, .json, .py, .csv)
            mime_type = IMAGE_EXT_MIME_MAP.get(ext)
            if not mime_type and content_type:
                if is_valid_multimodal_mime(content_type):
                    mime_type = content_type

            if not is_valid_multimodal_mime(mime_type):
                return None

            # If presigned URL
            if data.get("url"):
                img_resp = client.get(data["url"])
                if img_resp.status_code == 200:
                    return img_resp.content, mime_type
            elif data.get("text_content") and ext == ".svg":
                return data["text_content"].encode("utf-8"), mime_type
    except Exception:
        pass
    return None


def fetch_http_image(url: str) -> Optional[Tuple[bytes, str]]:
    """Fetches image bytes and MIME type from direct HTTP/HTTPS URL."""
    try:
        import httpx

        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                ext = Path(url.split("?")[0]).suffix.lower()
                mime_type = IMAGE_EXT_MIME_MAP.get(ext) or ct
                if not is_valid_multimodal_mime(mime_type):
                    return None
                return resp.content, mime_type
    except Exception:
        pass
    return None


def resolve_image_source(src: str) -> Optional[Tuple[bytes, str]]:
    """Resolves an image source (Base64 data URI, local file path, cloud:// file ID, or HTTP URL) to (raw_bytes, mime_type)."""
    src = (src or "").strip().strip("\"'")
    if not src:
        return None

    # 1. Base64 Data URI (e.g. data:image/png;base64,iVBORw0KGgo...)
    if src.startswith("data:") and ";base64," in src:
        try:
            mime_part, b64_part = src.split(";base64,", 1)
            mime_type = mime_part.replace("data:", "").strip().lower()
            if not is_valid_multimodal_mime(mime_type):
                mime_type = "image/png"
            raw_bytes = base64.b64decode(b64_part.strip())
            return raw_bytes, mime_type
        except Exception:
            return None

    # 2. HTTP/HTTPS URL
    if src.startswith("http://") or src.startswith("https://"):
        return fetch_http_image(src)

    # 3. Cloud / VFS (cloud://file_... or raw file_... ID)
    if src.startswith("cloud://") or (src.startswith("file_") and not Path(os.path.expanduser(src)).exists()):
        return fetch_vfs_image(src)

    # 4. Local File Path
    try:
        clean_path = os.path.expanduser(src)
        p = Path(clean_path)
        ext = p.suffix.lower()
        if ext in IMAGE_EXT_MIME_MAP and p.exists() and p.is_file():
            mime_type = IMAGE_EXT_MIME_MAP[ext]
            raw_bytes = p.read_bytes()
            return raw_bytes, mime_type
    except Exception:
        pass

    return None


def extract_and_resolve_images(message: Message) -> List[Tuple[bytes, str]]:
    """Extracts and loads binary image data and MIME types from message attachments or prompt tags."""
    resolved: List[Tuple[bytes, str]] = []
    seen_sources = set()

    sources: List[str] = []
    if getattr(message, "images", None):
        sources.extend(message.images)

    # Scan text content for attachment patterns
    text = message.content or message.body or ""
    if text:
        # Pattern 1: [Attached Image: /path/to/img.png] or [Image: /path/to/img.jpg] or [Screenshot: /path/to/img.png] or [Attached Image: cloud://file_123]
        for m in re.finditer(r"\[(?:Attached Image|Image|Screenshot):\s*([^\]]+)\]", text, re.IGNORECASE):
            sources.append(m.group(1).strip())

        # Pattern 2: Markdown image ![alt](/path/to/img.png)
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            sources.append(m.group(1).strip())

        # Pattern 3: Cloud VFS markdown links [name.jpg](cloud://file_123)
        for m in re.finditer(r"\[[^\]]*\]\((cloud://[^)]+)\)", text):
            sources.append(m.group(1).strip())

        # Pattern 4: Standalone cloud://file_... URI
        for m in re.finditer(r"\b(cloud://file_[a-zA-Z0-9_-]+)\b", text):
            sources.append(m.group(1).strip())

        # Pattern 5: Standalone local filesystem paths (e.g. pasted or drag-dropped into terminal)
        for line in text.splitlines():
            cleaned_line = line.strip().strip("\"'")
            if cleaned_line.startswith(("/", "~")):
                try:
                    p = Path(os.path.expanduser(cleaned_line))
                    if p.suffix.lower() in IMAGE_EXT_MIME_MAP and p.is_file():
                        sources.append(str(p))
                except Exception:
                    pass

    for src in sources:
        if src in seen_sources:
            continue
        seen_sources.add(src)

        try:
            res = resolve_image_source(src)
            if res and is_valid_multimodal_mime(res[1]):
                resolved.append(res)
        except Exception:
            pass

    return resolved


class BaseProvider(ABC):
    """Abstract interface for LLM provider adapters."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        max_delay: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("VALSTORM_MAX_RETRIES", "3"))
        )
        self.initial_delay = (
            initial_delay
            if initial_delay is not None
            else float(os.environ.get("VALSTORM_RETRY_INITIAL_DELAY", "1.0"))
        )
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else float(os.environ.get("VALSTORM_RETRY_BACKOFF_FACTOR", "2.0"))
        )
        self.max_delay = (
            max_delay
            if max_delay is not None
            else float(os.environ.get("VALSTORM_RETRY_MAX_DELAY", "30.0"))
        )
        self.extra_config = kwargs
        self._request_context: Dict[str, Any] = {}

    def set_request_context(self, **context: Any) -> None:
        """Attach per-run metadata (e.g. chat_id) that providers may forward upstream.

        Only the Valstorm gateway uses it today (X-Valstorm-Chat-Id, for per-chat metered billing);
        other providers ignore it.
        """
        self._request_context = {k: v for k, v in context.items() if v is not None}

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'openai', 'anthropic')."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageMetadata]:
        """Generate response non-streaming."""
        pass

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[StreamEvent, Tuple[Message, UsageMetadata]]]:
        """Default streaming fallback that calls generate and yields chunks/final result."""
        msg, usage = await self.generate(messages, tools=tools, model=model, **kwargs)
        if msg.content:
            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=msg.content)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
        yield (msg, usage)
