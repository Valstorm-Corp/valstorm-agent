"""AI Image & Video Generation Tools for Valstorm Agent Runtime.

Provides:
- generate_image: Proxies image generation (`gemini-3-pro-image`, `gemini-3.1-flash-image`)
  through `/v1/ai/media/generate-image` using JWT or `pat_...` auth, optionally saves to VFS
  and/or local `output_path`.
- generate_video: Proxies Veo 3.1 video generation (`veo-3.1-generate-001`,
  `veo-3.1-fast-generate-001`, `veo-3.1-lite-generate-001`) with optional `first_frame` and
  `last_frame` keyframes through `/v1/ai/media/generate-video`, polls
  `/v1/ai/media/operations/{operation_id}` until completion, and saves to VFS and/or local `output_path`.
"""

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import httpx

from core.tools import ToolRegistry, tool
from tools.valstorm_client import ValstormApiClient, resolve_valstorm_auth_context

_MEDIA_CLIENT_OVERRIDE: Optional[Any] = None


def set_media_client_override(client: Optional[Any]) -> None:
    """Sets a client override for unit testing media generation tools."""
    global _MEDIA_CLIENT_OVERRIDE
    _MEDIA_CLIENT_OVERRIDE = client


def _resolve_client(override_client: Optional[Any] = None) -> ValstormApiClient:
    if override_client is not None:
        return override_client
    if _MEDIA_CLIENT_OVERRIDE is not None:
        return _MEDIA_CLIENT_OVERRIDE

    token, base_url, refresh_token, auth_file_path = resolve_valstorm_auth_context()
    if not token:
        raise ValueError(
            "No Valstorm authentication credentials found (JWT or pat_...). "
            "Authenticate via `valstorm login` or set VALSTORM_API_KEY / VALSTORM_ACCESS_TOKEN."
        )
    return ValstormApiClient(
        token=token,
        base_url=base_url,
        refresh_token=refresh_token,
        auth_file_path=auth_file_path,
    )


def _normalize_frame_or_reference_input(item: Any) -> Any:
    """
    Converts a local disk file path into a `data:<mime>;base64,...` URI while preserving
    VFS `file_id` strings (`file_...`, `cloud://file_...`), existing data URIs, and raw base64.
    """
    if not item:
        return None
    if isinstance(item, dict):
        return item
    if isinstance(item, bytes):
        return f"data:image/png;base64,{base64.b64encode(item).decode('ascii')}"
    if isinstance(item, str):
        clean = item.strip()
        if not clean:
            return None
        if clean.startswith(("file_", "cloud://", "data:")):
            return clean
        p = Path(clean).expanduser()
        if p.exists() and p.is_file():
            raw = p.read_bytes()
            mime, _ = mimetypes.guess_type(str(p))
            mime = mime or "image/png"
            b64 = base64.b64encode(raw).decode("ascii")
            return f"data:{mime};base64,{b64}"
        return clean
    return item


async def _write_media_output(
    output_path: Optional[str],
    base64_payload: Optional[str],
    download_url: Optional[str],
    client: Any,
) -> Optional[str]:
    if not output_path:
        return None
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if base64_payload:
        out_file.write_bytes(base64.b64decode(base64_payload))
        return str(out_file)

    if download_url:
        if download_url.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http_cli:
                resp = await http_cli.get(download_url)
                resp.raise_for_status()
                out_file.write_bytes(resp.content)
                return str(out_file)
        elif hasattr(client, "_request_with_retry"):
            resp = await client._request_with_retry("GET", download_url)
            out_file.write_bytes(resp.content)
            return str(out_file)

    return None


@tool
async def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model: str = "gemini-3-pro-image",
    aspect_ratio: str = "16:9",
    resolution: str = "1k",
    reference_images: Optional[List[str]] = None,
    save_to_vfs: bool = False,
    vault_id: Optional[str] = None,
    filename: Optional[str] = None,
    is_public: bool = False,
) -> str:
    """Generates an AI image using Gemini 3 Pro Image (`gemini-3-pro-image`) or Gemini 3.1 Flash Image (`gemini-3.1-flash-image`).

    Routes through the authenticated Valstorm backend (`/v1/ai/media/generate-image`) using the active
    JWT or `pat_...` Personal Access Token, meters usage against the organization ledger, and saves
    the resulting `.png` to `output_path` on local disk and/or to Cloud VFS (`save_to_vfs=True`).

    Args:
        prompt: Detailed visual description of the image to generate.
        output_path: Optional local file path to save the generated `.png` (e.g. 'plates/hero_16x9.png').
        model: Image model ID ('gemini-3-pro-image' or 'gemini-3.1-flash-image'). Defaults to 'gemini-3-pro-image'.
        aspect_ratio: Target aspect ratio ('16:9', '9:16', '1:1', '4:3', '3:4'). Defaults to '16:9'.
        resolution: Target resolution tier ('0.5k', '1k', '2k', '4k'). Defaults to '1k'.
        reference_images: Optional list of reference images (local file paths, VFS 'file_...' IDs, or base64 strings).
        save_to_vfs: If True, automatically persists the generated `.png` into Cloud VFS.
        vault_id: Optional VFS Vault ID (e.g. 'vaul_123') when `save_to_vfs=True`.
        filename: Optional filename in VFS (e.g. 'hero_keyframe.png') when `save_to_vfs=True`.
        is_public: If True and `save_to_vfs=True`, marks the VFS file as public.

    Returns:
        JSON string containing status, model, local `output_path`, VFS `file_id`, `download_url`, `size_bytes`, and `cost_usd`.
    """
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        return "Error: 'prompt' is required for generate_image."

    try:
        client = _resolve_client()
        normalized_refs: Optional[List[Any]] = None
        if reference_images:
            normalized_refs = [
                r for r in (_normalize_frame_or_reference_input(item) for item in reference_images) if r is not None
            ]

        res = await client.ai_generate_image(
            prompt=clean_prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            reference_images=normalized_refs,
            save_to_vfs=save_to_vfs,
            vault_id=vault_id,
            filename=filename,
            is_public=is_public,
        )

        img_b64 = res.get("image_base64") or res.get("b64_json")
        dl_url = res.get("download_url")
        saved_local_path = await _write_media_output(output_path, img_b64, dl_url, client)

        summary = {
            "status": res.get("status", "completed"),
            "model": res.get("model", model),
            "aspect_ratio": res.get("aspect_ratio", aspect_ratio),
            "resolution": res.get("resolution", resolution),
            "mime_type": res.get("mime_type", "image/png"),
            "size_bytes": res.get("size_bytes", 0),
            "output_path": saved_local_path,
            "file_id": res.get("file_id"),
            "download_url": dl_url,
            "filename": res.get("filename"),
            "vault_id": res.get("vault_id"),
            "cost_usd": res.get("cost_usd", 0.0),
        }
        return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error generating image: {type(e).__name__}: {e}"


@tool
async def generate_video(
    prompt: str,
    output_path: Optional[str] = None,
    model: str = "veo-3.1-generate-001",
    aspect_ratio: str = "16:9",
    resolution: str = "1080p",
    duration_seconds: int = 8,
    first_frame: Optional[str] = None,
    last_frame: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    save_to_vfs: bool = False,
    vault_id: Optional[str] = None,
    filename: Optional[str] = None,
    is_public: bool = False,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 300.0,
) -> str:
    """Generates an AI video using Veo 3.1 (`veo-3.1-generate-001`, `veo-3.1-fast-generate-001`, `veo-3.1-lite-generate-001`).

    Routes through the authenticated Valstorm backend (`/v1/ai/media/generate-video`) with optional
    `first_frame` and `last_frame` keyframe interpolation (from local disk paths, VFS `file_id`s, or base64),
    polls `/v1/ai/media/operations/{operation_id}` until completion, and saves the `.mp4` to `output_path`
    and/or Cloud VFS (`save_to_vfs=True`).

    Args:
        prompt: Detailed visual and motion prompt for the video scene.
        output_path: Optional local file path to save the generated `.mp4` (e.g. 'plates/scene_01.mp4').
        model: Video model ID ('veo-3.1-generate-001', 'veo-3.1-fast-generate-001', 'veo-3.1-lite-generate-001').
        aspect_ratio: Video aspect ratio ('16:9' or '9:16'). Defaults to '16:9'.
        resolution: Video resolution ('1080p' or '720p'). Defaults to '1080p'.
        duration_seconds: Video duration in seconds (4, 6, or 8). Defaults to 8.
        first_frame: Optional starting keyframe (local image path, VFS 'file_...' ID, or base64 string).
        last_frame: Optional ending keyframe (local image path, VFS 'file_...' ID, or base64 string).
        negative_prompt: Optional negative prompt describing elements to avoid.
        save_to_vfs: If True, automatically persists the generated `.mp4` into Cloud VFS.
        vault_id: Optional VFS Vault ID (e.g. 'vaul_123') when `save_to_vfs=True`.
        filename: Optional filename in VFS (e.g. 'scene_01.mp4') when `save_to_vfs=True`.
        is_public: If True and `save_to_vfs=True`, marks the VFS file as public.
        poll_interval_seconds: Seconds between LRO status polls. Defaults to 2.0.
        timeout_seconds: Maximum seconds to wait for LRO video completion. Defaults to 300.0.

    Returns:
        JSON string containing `operation_id`, `status`, `model`, `duration_seconds`, local `output_path`,
        VFS `file_id`, `download_url`, `size_bytes`, and `cost_usd`.
    """
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        return "Error: 'prompt' is required for generate_video."

    try:
        client = _resolve_client()
        norm_first = _normalize_frame_or_reference_input(first_frame)
        norm_last = _normalize_frame_or_reference_input(last_frame)

        op_res = await client.ai_generate_video(
            prompt=clean_prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=int(duration_seconds),
            first_frame=norm_first,
            last_frame=norm_last,
            negative_prompt=negative_prompt,
            save_to_vfs=save_to_vfs,
            vault_id=vault_id,
            filename=filename,
            is_public=is_public,
            wait=False,
        )

        operation_id = str(op_res.get("operation_id") or "")
        final_res = op_res

        if operation_id and not final_res.get("done"):
            deadline = time.monotonic() + float(timeout_seconds)
            while time.monotonic() < deadline:
                final_res = await client.ai_get_media_operation(operation_id)
                if isinstance(final_res, dict) and final_res.get("done"):
                    break
                await asyncio.sleep(max(0.01, float(poll_interval_seconds)))

        if isinstance(final_res, dict) and final_res.get("status") == "FAILED":
            return f"Error: Video generation operation {operation_id} failed: {final_res.get('error', 'Unknown error')}"

        vid_b64 = final_res.get("video_base64")
        dl_url = final_res.get("download_url")
        saved_local_path = await _write_media_output(output_path, vid_b64, dl_url, client)

        summary = {
            "operation_id": operation_id,
            "status": final_res.get("status", "COMPLETED"),
            "done": bool(final_res.get("done", True)),
            "model": final_res.get("model", model),
            "aspect_ratio": final_res.get("aspect_ratio", aspect_ratio),
            "resolution": final_res.get("resolution", resolution),
            "duration_seconds": final_res.get("duration_seconds", duration_seconds),
            "first_frame_used": bool(final_res.get("first_frame_used", norm_first is not None)),
            "last_frame_used": bool(final_res.get("last_frame_used", norm_last is not None)),
            "mime_type": final_res.get("mime_type", "video/mp4"),
            "size_bytes": final_res.get("size_bytes", 0),
            "output_path": saved_local_path,
            "file_id": final_res.get("file_id"),
            "download_url": dl_url,
            "filename": final_res.get("filename"),
            "vault_id": final_res.get("vault_id"),
            "cost_usd": final_res.get("cost_usd", 0.0),
        }
        return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error generating video: {type(e).__name__}: {e}"


def create_media_generation_tools(client: Optional[Any] = None) -> List[Any]:
    """Returns list of media generation tool functions."""
    if client is not None:
        set_media_client_override(client)
    return [generate_image, generate_video]


def register_media_generation_tools(
    registry: ToolRegistry,
    client: Optional[Any] = None,
) -> ToolRegistry:
    """Registers generate_image and generate_video into the provided ToolRegistry."""
    for tool_fn in create_media_generation_tools(client=client):
        registry.register(tool_fn)
    return registry
