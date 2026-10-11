"""Local secret redaction and LLM egress sanitizer.

Prevents sensitive API keys, auth tokens, database credentials, environment secrets,
and private keys on developer machines from leaking to cloud LLM context windows.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Pattern, Set, Tuple

# Sensitive environment key fragments to discover and mask
SENSITIVE_ENV_SUBSTRINGS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "DATABASE_URL",
    "MONGO_URI",
    "REDIS_URL",
    "PRIVATE",
    "CREDENTIAL",
    "WEBHOOK",
    "BEARER",
    "SALT",
    "ENCRYPT",
    "SIGNING",
)

# High-entropy known service token & credential regexes
SECRET_PATTERNS: List[Tuple[Pattern[str], str]] = [
    # Generic .env line key-value matching: FOO_KEY=bar, JWT_SECRET="...", export DB_PASSWORD=...
    (
        re.compile(
            r"""(?m)^([ \t]*(?:export[ \t]+)?([A-Za-z0-9_]*?(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|AUTH|DATABASE_URL|MONGO_URI|REDIS_URL|PRIVATE|CREDENTIAL|WEBHOOK|BEARER|SALT|ENCRYPT|SIGNING)[A-Za-z0-9_]*?)[ \t]*=[ \t]*)(['"]?)([^\r\n"'\s]{6,})\3""",
            re.IGNORECASE,
        ),
        r"\1\3<REDACTED_ENV_SECRET:\2>\3",
    ),
    # Private Keys & Certificates
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
            re.MULTILINE,
        ),
        "<REDACTED_PRIVATE_KEY_BLOCK>",
    ),
    (
        re.compile(
            r"-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----",
            re.MULTILINE,
        ),
        "<REDACTED_CERTIFICATE_BLOCK>",
    ),
    # OpenAI & Project Keys
    (
        re.compile(r"\b(?:sk-proj|sk-org|sk-admin)-[a-zA-Z0-9_\-]{20,}\b"),
        "<REDACTED_OPENAI_KEY>",
    ),
    (
        re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
        "<REDACTED_OPENAI_KEY>",
    ),
    # Anthropic Keys
    (
        re.compile(r"\bsk-ant-[a-zA-Z0-9_\-]{20,}\b"),
        "<REDACTED_ANTHROPIC_KEY>",
    ),
    # Google Cloud / Gemini API Keys
    (
        re.compile(r"\bAIzaSy[A-Za-z0-9_\-]{33}\b"),
        "<REDACTED_GOOGLE_API_KEY>",
    ),
    # GitHub Tokens
    (
        re.compile(r"\bghp_[a-zA-Z0-9]{36,}\b"),
        "<REDACTED_GITHUB_PAT>",
    ),
    (
        re.compile(r"\bgithub_pat_[a-zA-Z0-9_]{60,}\b"),
        "<REDACTED_GITHUB_FINE_GRAINED_PAT>",
    ),
    (
        re.compile(r"\bgh[ousr]_[a-zA-Z0-9]{36,}\b"),
        "<REDACTED_GITHUB_TOKEN>",
    ),
    # Slack Tokens & Webhooks
    (
        re.compile(r"\bxox[baprs]-[0-9]{10,14}-[0-9]{10,14}[a-zA-Z0-9_\-]*\b"),
        "<REDACTED_SLACK_TOKEN>",
    ),
    (
        re.compile(r"https:\/\/hooks\.slack\.com\/services\/T[0-9A-Z]+\/B[0-9A-Z]+\/[0-9a-zA-Z]+"),
        "<REDACTED_SLACK_WEBHOOK>",
    ),
    # AWS Credentials
    (
        re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
        "<REDACTED_AWS_ACCESS_KEY_ID>",
    ),
    # Stripe API Keys
    (
        re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[0-9a-zA-Z]{24,}\b"),
        "<REDACTED_STRIPE_KEY>",
    ),
    # Twilio API Key
    (
        re.compile(r"\bSK[0-9a-fA-F]{32}\b"),
        "<REDACTED_TWILIO_API_KEY>",
    ),
    # SendGrid API Key
    (
        re.compile(r"\bSG\.[a-zA-Z0-9_\-\.]{60,}\b"),
        "<REDACTED_SENDGRID_API_KEY>",
    ),
    # Database URIs with credentials: mongodb://user:pass@host, postgresql://..., redis://...
    (
        re.compile(
            r"(\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp(?:s)?):\/\/[^:\/\s]+:)([^@\s]+)(@)",
            re.IGNORECASE,
        ),
        r"\1<REDACTED_DB_PASSWORD>\3",
    ),
    # JSON Web Tokens (JWT)
    (
        re.compile(r"\beyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\b"),
        "<REDACTED_JWT_TOKEN>",
    ),
    # Explicit Bearer Token headers
    (
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([a-zA-Z0-9_\-\.]{20,})"),
        r"\1<REDACTED_BEARER_TOKEN>",
    ),
    # Key-value assignment config matching: api_key = "...", secret_key = "..."
    (
        re.compile(
            r"""(?i)(["']?(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)["']?\s*[:=]\s*["'])([^"'\r\n\s]{12,})(["'])"""
        ),
        r"\1<REDACTED_SECRET>\3",
    ),
]

IGNORED_ENV_VALUES = {
    "true",
    "false",
    "none",
    "null",
    "1",
    "0",
    "dev",
    "prod",
    "development",
    "production",
    "staging",
    "test",
    "localhost",
    "127.0.0.1",
    "default",
    "bearer",
    "basic",
    "undefined",
    "valstorm",
}


class SecretSanitizer:
    """High-performance in-memory sanitizer matching known secret signatures and loaded env variables."""

    def __init__(self, include_env: bool = True, custom_secrets: Optional[Dict[str, str]] = None) -> None:
        self.include_env = include_env
        self._exact_replacements: List[Tuple[str, str]] = []
        self._custom_secrets: Dict[str, str] = custom_secrets or {}
        self._reload_secrets()

    def _reload_secrets(self) -> None:
        """Collects sensitive environment variables and prepares exact replacement pairs sorted longest-first."""
        pairs: List[Tuple[str, str]] = []
        seen_vals: Set[str] = set()

        # Add custom secrets
        for name, val in self._custom_secrets.items():
            if val and len(val) >= 6:
                clean_val = val.strip()
                if clean_val not in seen_vals:
                    pairs.append((clean_val, f"<REDACTED_CUSTOM_SECRET:{name}>"))
                    seen_vals.add(clean_val)

        # Scan os.environ if enabled
        if self.include_env:
            for env_key, env_val in os.environ.items():
                if not env_val:
                    continue
                k_upper = env_key.upper()
                if any(sub in k_upper for sub in SENSITIVE_ENV_SUBSTRINGS):
                    clean_val = env_val.strip()
                    # Filter out short or generic values
                    if len(clean_val) < 6:
                        continue
                    if clean_val.lower() in IGNORED_ENV_VALUES:
                        continue
                    if clean_val not in seen_vals:
                        pairs.append((clean_val, f"<REDACTED_ENV_SECRET:{env_key}>"))
                        seen_vals.add(clean_val)

        # Sort longest secret first so subsets don't break larger tokens
        pairs.sort(key=lambda item: len(item[0]), reverse=True)
        self._exact_replacements = pairs

    def add_secret(self, name: str, value: str) -> None:
        """Dynamically register a sensitive secret value to redact."""
        if value and len(value) >= 6:
            self._custom_secrets[name] = value
            self._reload_secrets()

    def load_env_file(self, file_path_or_content: str | Path) -> None:
        """Parses a .env file or text content and registers all sensitive key/value pairs."""
        content = ""
        if isinstance(file_path_or_content, (str, Path)) and os.path.exists(str(file_path_or_content)):
            try:
                with open(file_path_or_content, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                return
        elif isinstance(file_path_or_content, str):
            content = file_path_or_content

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            parts = line.split("=", 1)
            key = parts[0].replace("export", "").strip()
            val = parts[1].strip().strip("'\"")
            if any(sub in key.upper() for sub in SENSITIVE_ENV_SUBSTRINGS):
                if len(val) >= 6 and val.lower() not in IGNORED_ENV_VALUES:
                    self._custom_secrets[key] = val
        self._reload_secrets()

    def sanitize(self, text: Any) -> str:
        """Sanitizes text, masking any detected API keys, credentials, or environment secrets.

        Executes in sub-millisecond time.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return ""

        result = text

        # 0. Fast Native Rust Regex Sanitize pass
        try:
            from core.native_bridge import native_sanitize_text
            native_res = native_sanitize_text(result)
            if native_res is not None:
                result = native_res
        except Exception:
            pass

        # 1. Exact Environment / Custom Secret Masking
        for secret_val, replacement_label in self._exact_replacements:
            if secret_val in result:
                result = result.replace(secret_val, replacement_label)

        # 2. Known API / Token Pattern Masking (for any custom or edge patterns)
        for pattern, replacement in SECRET_PATTERNS:
            result = pattern.sub(replacement, result)

        return result


# Global singleton instance
_GLOBAL_SANITIZER = SecretSanitizer()


def sanitize_text(text: Any) -> str:
    """Convenience helper to sanitize text using the global SecretSanitizer."""
    return _GLOBAL_SANITIZER.sanitize(text)


def register_secret_to_redact(name: str, value: str) -> None:
    """Registers a secret to be masked globally."""
    _GLOBAL_SANITIZER.add_secret(name, value)
