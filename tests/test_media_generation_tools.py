"""Unit tests for NF-520 AI Image & Video Generation Tools in apps/agent-runtime."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from core.context import load_profile
from tools import (
    create_tier1_tools,
    generate_image,
    generate_video,
    set_media_client_override,
)
from tools.valstorm_platform_client import RemotePlatformContext

_SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_SAMPLE_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom\x00\x00\x00\x08free"


class FakeMediaValstormClient:
    """Fake ValstormApiClient supporting JWT and pat_... auth, image gen, and LRO video polling."""

    def __init__(self, token: str = "pat_test_token_123"):
        self.token = token
        self.base_url = "http://localhost:8010/v1"
        self.image_calls: List[Dict[str, Any]] = []
        self.video_calls: List[Dict[str, Any]] = []
        self.poll_calls: List[str] = []
        self._poll_count = 0

    async def ai_generate_image(
        self,
        prompt: str,
        model: str = "gemini-3-pro-image",
        aspect_ratio: str = "16:9",
        resolution: str = "1k",
        reference_images: Optional[List[Any]] = None,
        save_to_vfs: bool = False,
        vault_id: Optional[str] = None,
        filename: Optional[str] = None,
        is_public: bool = False,
    ) -> Dict[str, Any]:
        self.image_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "reference_images": reference_images,
                "save_to_vfs": save_to_vfs,
                "vault_id": vault_id,
                "filename": filename,
                "token": self.token,
            }
        )
        b64 = base64.b64encode(_SAMPLE_PNG_BYTES).decode("ascii")
        return {
            "status": "completed",
            "model": model,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "mime_type": "image/png",
            "image_base64": b64,
            "b64_json": b64,
            "size_bytes": len(_SAMPLE_PNG_BYTES),
            "file_id": "file_img_vfs_99" if save_to_vfs else None,
            "download_url": "https://cdn.valstorm.com/file_img_vfs_99.png" if save_to_vfs else None,
            "filename": filename or "generated.png",
            "vault_id": vault_id,
            "cost_usd": 0.134,
        }

    async def ai_generate_video(
        self,
        prompt: str,
        model: str = "veo-3.1-generate-001",
        aspect_ratio: str = "16:9",
        resolution: str = "1080p",
        duration_seconds: int = 8,
        first_frame: Optional[Any] = None,
        last_frame: Optional[Any] = None,
        negative_prompt: Optional[str] = None,
        save_to_vfs: bool = False,
        vault_id: Optional[str] = None,
        filename: Optional[str] = None,
        is_public: bool = False,
        wait: bool = False,
    ) -> Dict[str, Any]:
        self.video_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration_seconds": duration_seconds,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "save_to_vfs": save_to_vfs,
                "vault_id": vault_id,
                "filename": filename,
                "token": self.token,
            }
        )
        return {
            "operation_id": "op_veo_test_777",
            "status": "RUNNING",
            "done": False,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "save_to_vfs": save_to_vfs,
            "vault_id": vault_id,
            "filename": filename,
        }

    async def ai_get_media_operation(self, operation_id: str) -> Dict[str, Any]:
        self.poll_calls.append(operation_id)
        self._poll_count += 1
        if self._poll_count < 2:
            return {
                "operation_id": operation_id,
                "status": "RUNNING",
                "done": False,
                "model": "veo-3.1-generate-001",
            }
        b64 = base64.b64encode(_SAMPLE_MP4_BYTES).decode("ascii")
        return {
            "operation_id": operation_id,
            "status": "COMPLETED",
            "done": True,
            "model": "veo-3.1-generate-001",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "duration_seconds": 8,
            "first_frame_used": True,
            "last_frame_used": True,
            "mime_type": "video/mp4",
            "video_base64": b64,
            "size_bytes": len(_SAMPLE_MP4_BYTES),
            "file_id": "file_vid_vfs_88",
            "download_url": "https://cdn.valstorm.com/file_vid_vfs_88.mp4",
            "filename": "nebula_finale.mp4",
            "vault_id": "vaul_123",
            "cost_usd": 3.20,
        }


@pytest.fixture(autouse=True)
def reset_media_client():
    set_media_client_override(None)
    yield
    set_media_client_override(None)


@pytest.mark.asyncio
async def test_generate_image_tool_local_output_and_vfs(tmp_path: Path):
    fake_client = FakeMediaValstormClient("pat_placeholder_test")
    set_media_client_override(fake_client)

    ref_img_path = tmp_path / "ref_logo.png"
    ref_img_path.write_bytes(_SAMPLE_PNG_BYTES)
    out_img_path = tmp_path / "out" / "hero_plate.png"

    res_str = await generate_image(
        prompt="Isometric glassmorphic CRM card with glowing indigo border",
        output_path=str(out_img_path),
        model="gemini-3-pro-image",
        aspect_ratio="16:9",
        reference_images=[str(ref_img_path), "file_existing_ref_1"],
        save_to_vfs=True,
        vault_id="vaul_creative_1",
        filename="hero_plate.png",
    )
    data = json.loads(res_str)
    assert data["status"] == "completed"
    assert data["file_id"] == "file_img_vfs_99"
    assert data["output_path"] == str(out_img_path.resolve())
    assert out_img_path.exists()
    assert out_img_path.read_bytes() == _SAMPLE_PNG_BYTES

    # Verify local reference image was base64-encoded while VFS file_id was preserved
    assert len(fake_client.image_calls) == 1
    sent_refs = fake_client.image_calls[0]["reference_images"]
    assert sent_refs[0].startswith("data:image/png;base64,")
    assert sent_refs[1] == "file_existing_ref_1"


@pytest.mark.asyncio
async def test_generate_video_tool_lro_polling_and_keyframes(tmp_path: Path):
    fake_client = FakeMediaValstormClient("jwt_placeholder_test")
    set_media_client_override(fake_client)

    first_frame_path = tmp_path / "keyframe_start.png"
    first_frame_path.write_bytes(_SAMPLE_PNG_BYTES)
    out_mp4_path = tmp_path / "renders" / "nebula_finale.mp4"

    res_str = await generate_video(
        prompt="3D volumetric neural nebula camera push-in",
        output_path=str(out_mp4_path),
        model="veo-3.1-generate-001",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_seconds=8,
        first_frame=str(first_frame_path),
        last_frame="file_end_keyframe_2",
        save_to_vfs=True,
        vault_id="vaul_123",
        filename="nebula_finale.mp4",
        poll_interval_seconds=0.01,
    )
    data = json.loads(res_str)
    assert data["operation_id"] == "op_veo_test_777"
    assert data["status"] == "COMPLETED"
    assert data["done"] is True
    assert data["file_id"] == "file_vid_vfs_88"
    assert data["output_path"] == str(out_mp4_path.resolve())
    assert out_mp4_path.exists()
    assert out_mp4_path.read_bytes() == _SAMPLE_MP4_BYTES
    assert len(fake_client.poll_calls) == 2


def test_remote_platform_context_ai_generate_image_and_video(tmp_path: Path):
    fake_client = FakeMediaValstormClient("pat_placeholder_sdk")
    platform = RemotePlatformContext(client=fake_client)

    img_out = tmp_path / "sdk_img.png"
    img_res = platform.ai.generate_image(
        prompt="Dark slate enterprise UI hero",
        model="gemini-3.1-flash-image",
        save_to_vfs=True,
        output_path=str(img_out),
    )
    assert img_res["status"] == "completed"
    assert img_out.exists()

    vid_out = tmp_path / "sdk_vid.mp4"
    vid_res = platform.ai.generate_video(
        prompt="Smooth pan across dashboard",
        model="veo-3.1-fast-generate-001",
        duration_seconds=6,
        poll_interval_seconds=0.01,
        output_path=str(vid_out),
    )
    assert vid_res["status"] == "COMPLETED"
    assert vid_out.exists()


def test_agent_profiles_whitelist_generate_image_and_generate_video():
    tier1_names = [getattr(t, "_tool_name", t.__name__) for t in create_tier1_tools()]
    assert "generate_image" in tier1_names
    assert "generate_video" in tier1_names

    for profile_slug in ("creative-director", "video-editor", "marketing-specialist", "software-engineer"):
        prof = load_profile(profile_slug)
        assert prof is not None, f"Profile {profile_slug} should load"
        allowed = prof.get("allowed_tools", [])
        assert "generate_image" in allowed, f"generate_image missing from {profile_slug}"
        assert "generate_video" in allowed, f"generate_video missing from {profile_slug}"
