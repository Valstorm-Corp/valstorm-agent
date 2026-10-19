"""Base provider abstract interface and multimodal vision helpers for agent runtime."""

from abc import ABC, abstractmethod
import base64
from collections import OrderedDict
import hashlib
import io
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
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

INLINE_IMAGE_EXT_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".heic": "image/heic",
}

MAX_INLINE_MEDIA_BYTES = int(os.environ.get("VALSTORM_MAX_INLINE_MEDIA_BYTES", str(15 * 1024 * 1024)))
COMPRESS_IMAGE_THRESHOLD_BYTES = int(
    os.environ.get("VALSTORM_COMPRESS_IMAGE_THRESHOLD_BYTES", str(512 * 1024))
)
IMAGE_CACHE_MAX_ITEMS = int(os.environ.get("VALSTORM_IMAGE_CACHE_MAX_ITEMS", "64"))
IMAGE_CACHE_TTL_SEC = float(os.environ.get("VALSTORM_IMAGE_CACHE_TTL_SEC", "1800"))

_IMAGE_CACHE_LOCK = threading.Lock()
_RESOLVED_IMAGE_CACHE: "OrderedDict[str, Tuple[bytes, str, float]]" = OrderedDict()
_IMAGE_FETCH_STATS: Dict[str, int] = {"hits": 0, "misses": 0, "compressed": 0}


def clear_image_cache() -> None:
    """Clears the in-memory resolved image cache and resets stats."""
    with _IMAGE_CACHE_LOCK:
        _RESOLVED_IMAGE_CACHE.clear()
        _IMAGE_FETCH_STATS["hits"] = 0
        _IMAGE_FETCH_STATS["misses"] = 0
        _IMAGE_FETCH_STATS["compressed"] = 0


def get_image_cache_stats() -> Dict[str, int]:
    """Returns current cache hit/miss/compression counters for diagnostics and testing."""
    with _IMAGE_CACHE_LOCK:
        return {
            **_IMAGE_FETCH_STATS,
            "entries": len(_RESOLVED_IMAGE_CACHE),
        }


def _optimize_inline_image_bytes(raw_bytes: bytes, mime_type: str) -> Tuple[bytes, str]:
    """Downsamples and compresses oversized still images (>512KB) to prevent multi-MB TLS stream fragmentation."""
    threshold = int(
        os.environ.get("VALSTORM_COMPRESS_IMAGE_THRESHOLD_BYTES", str(COMPRESS_IMAGE_THRESHOLD_BYTES))
    )
    if threshold <= 0 or len(raw_bytes) <= threshold:
        return raw_bytes, mime_type
    norm_mime = (mime_type or "").strip().lower()
    if norm_mime not in ("image/png", "image/jpeg", "image/webp", "image/bmp"):
        return raw_bytes, mime_type

    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(raw_bytes)) as img:
            max_dim = int(os.environ.get("VALSTORM_MAX_INLINE_IMAGE_DIMENSION", "1920"))
            width, height = img.size
            if max(width, height) > max_dim:
                scale = max_dim / float(max(width, height))
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                img = img.resize(new_size, resample)

            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.convert("RGBA")
                bg.paste(alpha, mask=alpha.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            out = io.BytesIO()
            img.save(out, format="JPEG", quality=82, optimize=True)
            compressed = out.getvalue()
            if len(compressed) < len(raw_bytes):
                with _IMAGE_CACHE_LOCK:
                    _IMAGE_FETCH_STATS["compressed"] += 1
                return compressed, "image/jpeg"
    except Exception:
        pass

    # Fallback 2: Native macOS /usr/bin/sips (available on all macOS hosts without Pillow)
    if os.path.exists("/usr/bin/sips"):
        import shutil
        import subprocess
        import tempfile

        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }
        in_ext = ext_map.get(norm_mime, ".png")
        tmpdir = tempfile.mkdtemp(prefix="vsagent_img_")
        try:
            in_path = os.path.join(tmpdir, f"in{in_ext}")
            out_path = os.path.join(tmpdir, "out.jpg")
            with open(in_path, "wb") as f:
                f.write(raw_bytes)
            max_dim = int(os.environ.get("VALSTORM_MAX_INLINE_IMAGE_DIMENSION", "1920"))
            needs_resize = False
            if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n" and len(raw_bytes) >= 24:
                import struct

                w, h = struct.unpack(">II", raw_bytes[16:24])
                if max(w, h) > max_dim:
                    needs_resize = True
            cmd = ["/usr/bin/sips"]
            if needs_resize:
                cmd.extend(["-Z", str(max_dim)])
            cmd.extend(["-s", "format", "jpeg", "-s", "formatOptions", "80", in_path, "--out", out_path])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
            )
            if proc.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    compressed = f.read()
                if 0 < len(compressed) < len(raw_bytes):
                    with _IMAGE_CACHE_LOCK:
                        _IMAGE_FETCH_STATS["compressed"] += 1
                    return compressed, "image/jpeg"
        except Exception:
            pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return raw_bytes, mime_type


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
                if img_resp.status_code == 200 and len(img_resp.content) <= MAX_INLINE_MEDIA_BYTES:
                    return img_resp.content, mime_type
            elif data.get("text_content") and ext == ".svg":
                raw_svg = data["text_content"].encode("utf-8")
                if len(raw_svg) <= MAX_INLINE_MEDIA_BYTES:
                    return raw_svg, mime_type
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
                if len(resp.content) > MAX_INLINE_MEDIA_BYTES:
                    return None
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
            if len(raw_bytes) > MAX_INLINE_MEDIA_BYTES:
                return None
            return _optimize_inline_image_bytes(raw_bytes, mime_type)
        except Exception:
            return None

    # Check in-memory LRU cache for VFS (cloud:// / file_) and HTTP/HTTPS sources
    cache_key: Optional[str] = None
    if src.startswith(("http://", "https://", "cloud://", "file_")):
        cache_key = f"remote:{src.replace('cloud://', '').strip()}" if src.startswith(("cloud://", "file_")) else f"url:{src}"
        now = time.monotonic()
        with _IMAGE_CACHE_LOCK:
            cached_entry = _RESOLVED_IMAGE_CACHE.get(cache_key)
            if cached_entry is not None:
                c_bytes, c_mime, c_ts = cached_entry
                if (now - c_ts) <= IMAGE_CACHE_TTL_SEC:
                    _RESOLVED_IMAGE_CACHE.move_to_end(cache_key)
                    _IMAGE_FETCH_STATS["hits"] += 1
                    return c_bytes, c_mime
                else:
                    _RESOLVED_IMAGE_CACHE.pop(cache_key, None)

    resolved_pair: Optional[Tuple[bytes, str]] = None

    # 2. HTTP/HTTPS URL
    if src.startswith("http://") or src.startswith("https://"):
        resolved_pair = fetch_http_image(src)

    # 3. Cloud / VFS (cloud://file_... or raw file_... ID)
    elif src.startswith("cloud://") or (src.startswith("file_") and not Path(os.path.expanduser(src)).exists()):
        resolved_pair = fetch_vfs_image(src)

    # 4. Local File Path
    else:
        try:
            clean_path = os.path.expanduser(src)
            p = Path(clean_path)
            ext = p.suffix.lower()
            if ext in IMAGE_EXT_MIME_MAP and p.exists() and p.is_file():
                stat = p.stat()
                if stat.st_size > MAX_INLINE_MEDIA_BYTES:
                    return None
                local_key = f"file:{p.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
                with _IMAGE_CACHE_LOCK:
                    cached_local = _RESOLVED_IMAGE_CACHE.get(local_key)
                    if cached_local is not None:
                        _RESOLVED_IMAGE_CACHE.move_to_end(local_key)
                        _IMAGE_FETCH_STATS["hits"] += 1
                        return cached_local[0], cached_local[1]
                cache_key = local_key
                mime_type = IMAGE_EXT_MIME_MAP[ext]
                raw_bytes = p.read_bytes()
                resolved_pair = (raw_bytes, mime_type)
        except Exception:
            pass

    if resolved_pair is not None:
        opt_bytes, opt_mime = _optimize_inline_image_bytes(resolved_pair[0], resolved_pair[1])
        if cache_key:
            with _IMAGE_CACHE_LOCK:
                _IMAGE_FETCH_STATS["misses"] += 1
                _RESOLVED_IMAGE_CACHE[cache_key] = (opt_bytes, opt_mime, time.monotonic())
                _RESOLVED_IMAGE_CACHE.move_to_end(cache_key)
                while len(_RESOLVED_IMAGE_CACHE) > IMAGE_CACHE_MAX_ITEMS:
                    _RESOLVED_IMAGE_CACHE.popitem(last=False)
        return opt_bytes, opt_mime

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

        # Pattern 5: Standalone local filesystem paths (e.g. pasted or drag-dropped into terminal).
        # Only auto-inline still images (INLINE_IMAGE_EXT_MIME_MAP); local video/audio/PDF paths
        # pasted in prompts are file paths for local CLI/editing tools and must not be Base64-inlined.
        for line in text.splitlines():
            cleaned_line = line.strip().strip("\"'")
            if cleaned_line.startswith(("/", "~")):
                try:
                    p = Path(os.path.expanduser(cleaned_line))
                    if p.suffix.lower() in INLINE_IMAGE_EXT_MIME_MAP and p.is_file():
                        sources.append(str(p))
                except Exception:
                    pass

    for src in sources:
        if src in seen_sources:
            continue
        seen_sources.add(src)

        try:
            # Never inline local video/audio files into chat messages; local daemon tools process them on disk.
            clean_src = (src or "").strip().strip("\"'")
            if clean_src and not clean_src.startswith(("data:", "http://", "https://", "cloud://", "file_")):
                local_ext = Path(os.path.expanduser(clean_src)).suffix.lower()
                if local_ext and local_ext not in INLINE_IMAGE_EXT_MIME_MAP and local_ext != ".pdf":
                    continue

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
