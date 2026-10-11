"""Lightweight Multimodal Vision & Image Analysis Tool for Valstorm Agent Runtime.

Provides:
- analyze_image: Analyzes local screenshots, UI mockups, and images using cloud multimodal
  models (Gemini, OpenAI, Anthropic) or macOS Apple Vision OCR without heavy local ML dependencies.
- vision_analyze: Alias for analyze_image.
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from core.keystore import KeyStore
from core.models import Message
from core.tools import ToolRegistry, tool
import tempfile
from providers.base import IMAGE_EXT_MIME_MAP, resolve_image_source

logger = logging.getLogger("vsagent.tools.vision")

_VISION_PROVIDER_OVERRIDE: Optional[Any] = None


def set_vision_provider_override(provider: Optional[Any]) -> None:
    """Sets a global provider override for testing vision execution."""
    global _VISION_PROVIDER_OVERRIDE
    _VISION_PROVIDER_OVERRIDE = provider


def _get_vision_provider(
    preferred_provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[Any], str, str]:
    """Resolves provider instance, provider name, and default vision model name."""
    if _VISION_PROVIDER_OVERRIDE is not None:
        return _VISION_PROVIDER_OVERRIDE, "custom", "custom-vision-model"

    keystore = KeyStore()

    order = []
    if preferred_provider:
        order.append(preferred_provider.lower())
    for p in ["gemini", "openai", "anthropic"]:
        if p not in order:
            order.append(p)

    for prov in order:
        resolved_key = keystore.get_api_key(prov, override_key=api_key if prov == preferred_provider else None)
        if resolved_key:
            if prov in ("gemini", "google"):
                from providers.gemini import GeminiProvider
                return GeminiProvider(api_key=resolved_key), "gemini", "gemini-flash-latest"
            elif prov in ("openai", "chatgpt"):
                from providers.openai import OpenAIProvider
                return OpenAIProvider(api_key=resolved_key), "openai", "gpt-4o"
            elif prov in ("anthropic", "claude"):
                from providers.anthropic import AnthropicProvider
                return AnthropicProvider(api_key=resolved_key), "anthropic", "claude-sonnet-4-5"

    return None, "", ""


def _extract_image_metadata(file_path: Path) -> Dict[str, Any]:
    """Extracts lightweight image metadata (dimensions, format, size) without heavy ML dependencies."""
    meta: Dict[str, Any] = {
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": file_path.stat().st_size,
    }
    size_kb = meta["size_bytes"] / 1024
    if size_kb < 1024:
        meta["size_str"] = f"{size_kb:.1f} KB"
    else:
        meta["size_str"] = f"{size_kb / 1024:.2f} MB"

    try:
        from PIL import Image as PILImage
        with PILImage.open(file_path) as img:
            meta["width"], meta["height"] = img.size
            meta["format"] = img.format
            meta["mode"] = img.mode
    except Exception:
        pass

    return meta


def _macos_native_ocr(image_path: Path) -> Optional[str]:
    """Uses macOS Apple Vision framework via swift CLI for zero-overhead local OCR."""
    if sys.platform != "darwin" or not shutil.which("swift"):
        return None

    resolved_escaped = str(image_path.resolve()).replace('"', '\\"')
    swift_script = f"""
import Foundation
import Vision
import AppKit

let url = URL(fileURLWithPath: "{resolved_escaped}")
guard let image = NSImage(contentsOf: url),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {{
    exit(1)
}}

let request = VNRecognizeTextRequest {{ req, err in
    guard let results = req.results as? [VNRecognizedTextObservation] else {{ return }}
    for obs in results {{
        guard let candidate = obs.topCandidates(1).first else {{ continue }}
        print(candidate.string)
    }}
}}
request.recognitionLevel = .accurate
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])
"""
    try:
        proc = subprocess.run(
            ["swift", "-e", swift_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception as e:
        logger.debug(f"macOS Swift Vision OCR failed: {e}")
    return None


@tool
async def analyze_image(
    path: str,
    prompt: Optional[str] = None,
    detail: Optional[str] = "high",
) -> str:
    """Inspects and visually analyzes an image file (business card, screenshot, diagram, UI mockup, photo, document) using multimodal vision.

    Supports:
      - Cloud VFS files: 'cloud://file_123' or 'file_123'
      - Local disk paths: '/path/to/image.png' or 'image.jpg'
      - Direct HTTP/HTTPS URLs: 'https://example.com/card.png'
      - Base64 data URIs: 'data:image/png;base64,...'

    Use this tool whenever you need to inspect an image from cloud VFS or disk, read text from a business card,
    diagnose visual UI bugs, read text/layout from a screenshot, or analyze charts and diagrams.

    Args:
        path: Path or URI to the image file (e.g. 'cloud://file_dxXdv2qwweaj6QFx', 'file_123', 'screenshot.png', 'ui_mockup.jpg').
        prompt: Specific question or instruction for analyzing the image (e.g. 'Extract all business card details: Name, Title, Company, Phone, Email, Address, Website').
        detail: Detail level ('high', 'low', or 'auto'). Defaults to 'high'.

    Returns:
        A rich visual analysis and complete OCR extraction of the image content.
    """
    clean_path = (path or "").strip()
    if not clean_path:
        return "Error: An image file path must be provided for analyze_image."

    target_path = Path(os.path.expanduser(clean_path)).resolve()
    temp_file_created = False
    temp_file_path = None

    if not target_path.is_file():
        # Attempt to resolve from cloud://, VFS file_id, HTTP(S) URL, or Base64
        resolved_img = resolve_image_source(clean_path)
        if not resolved_img:
            return f"Error: Image file not found at '{clean_path}' (resolved to '{target_path}')."

        raw_bytes, mime_type = resolved_img
        # Infer extension from mime
        ext = ".jpg"
        for e, m in IMAGE_EXT_MIME_MAP.items():
            if m == mime_type:
                ext = e
                break

        try:
            tmp_fd, tmp_name = tempfile.mkstemp(prefix="vs_vision_", suffix=ext)
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(raw_bytes)
            target_path = Path(tmp_name).resolve()
            temp_file_created = True
            temp_file_path = tmp_name
        except Exception as e:
            return f"Error: Could not write remote image buffer to temporary file: {e}"

    ext = target_path.suffix.lower()
    if ext not in IMAGE_EXT_MIME_MAP:
        if temp_file_created and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        supported = ", ".join(IMAGE_EXT_MIME_MAP.keys())
        return f"Error: Unsupported image extension '{ext}'. Supported formats: {supported}."

    try:
        meta = await asyncio.to_thread(_extract_image_metadata, target_path)
        dim_str = f" | {meta['width']}x{meta['height']} px" if "width" in meta and "height" in meta else ""
        header_meta = f"{meta['filename']} ({meta.get('format', ext.lstrip('.').upper())}{dim_str} | {meta['size_str']})"

        instruction = (prompt or "").strip()
        if not instruction:
            instruction = (
                "Analyze this image and describe what you see in detail, including visual layout, "
                "UI components, text content, colors, and any visual defects or errors."
            )

        # Multimodal LLM Vision Call
        provider_inst, prov_name, default_model = _get_vision_provider()

        if provider_inst is not None:
            try:
                msg = Message(
                    role="user",
                    content=instruction,
                    images=[str(target_path)],
                )
                if hasattr(provider_inst, "generate"):
                    res = await provider_inst.generate([msg], model=default_model)
                    response_msg = res[0] if isinstance(res, tuple) else res
                elif hasattr(provider_inst, "generate_response"):
                    res = await provider_inst.generate_response([msg], model=default_model)
                    response_msg = res[0] if isinstance(res, tuple) else res
                else:
                    raise AttributeError(f"Provider {prov_name} has no generate method")
                body = getattr(response_msg, "content", None) or str(response_msg)

                return f"🖼️ [Visual Analysis - {prov_name.capitalize()} Vision]\nFile: {header_meta}\n\n{body}"
            except Exception as e:
                logger_err = str(e)
        else:
            logger_err = "No cloud multimodal API key configured (GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)."

        # Fallback to macOS Apple Vision OCR
        ocr_text = await asyncio.to_thread(_macos_native_ocr, target_path)
        if ocr_text:
            return (
                f"🖼️ [Visual Inspection - macOS Apple Vision Native OCR]\n"
                f"File: {header_meta}\n"
                f"Note: Cloud multimodal inference unavailable ({logger_err}). Extracted on-device OCR text:\n\n"
                f"--- Extracted Text ---\n"
                f"{ocr_text}"
            )

        return (
            f"🖼️ [Image Metadata]\n"
            f"File: {header_meta}\n"
            f"Path: {target_path}\n\n"
            f"Note: Could not run visual inference: {logger_err}"
        )
    finally:
        if temp_file_created and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@tool
async def vision_analyze(
    path: str,
    prompt: Optional[str] = None,
    detail: Optional[str] = "high",
) -> str:
    """Inspects and visually analyzes an image file (screenshot, business card, diagram, UI mockup). Alias for analyze_image.

    Supports:
      - Cloud VFS files: 'cloud://file_123' or 'file_123'
      - Local disk paths: '/path/to/image.png' or 'image.jpg'
      - Direct HTTP/HTTPS URLs: 'https://example.com/card.png'
      - Base64 data URIs: 'data:image/png;base64,...'

    Args:
        path: Path or URI to the image file (e.g. 'cloud://file_dxXdv2qwweaj6QFx', 'file_123', 'screenshot.png').
        prompt: Specific question or instruction for analyzing the image.
        detail: Detail level ('high', 'low', or 'auto'). Defaults to 'high'.

    Returns:
        A rich visual analysis and OCR extraction of the image.
    """
    return await analyze_image(path=path, prompt=prompt, detail=detail)


def create_vision_tools() -> List[Any]:
    """Returns the list of vision analysis tools."""
    return [analyze_image, vision_analyze]


def register_vision_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers vision analysis tools into the provided ToolRegistry."""
    for tool_fn in create_vision_tools():
        registry.register(tool_fn)
    return registry
