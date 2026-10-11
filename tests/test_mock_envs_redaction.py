"""Automated test verifying secret redaction against mock .env variations."""

from pathlib import Path
from core.sanitizer import SecretSanitizer
from core.tools import ToolRegistry


def test_all_mock_env_files_are_redacted():
    mock_env_dir = Path(__file__).resolve().parent / "mock_envs"
    assert mock_env_dir.exists()

    env_files = list(mock_env_dir.glob(".env*"))
    assert len(env_files) >= 3

    sanitizer = SecretSanitizer(include_env=True)

    for env_path in env_files:
        raw_text = env_path.read_text(encoding="utf-8")
        sanitizer.load_env_file(env_path)
        sanitized_text = sanitizer.sanitize(raw_text)

        lines = raw_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.replace("export", "").strip()
            v = v.strip().strip("'\"")
            if len(v) >= 6 and v.lower() not in {"development", "production", "staging", "localhost", "127.0.0.1", "true", "false", "none", "null"}:
                assert v not in sanitized_text, f"Secret {k} leaked in {env_path.name}!"
