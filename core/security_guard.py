"""Perimeter Defense, Quarantine & Malware Neutralization Guard for Valstorm Agent Runtime.

Provides:
- SecurityGuard: Hardened deterministic filters for inbound requests, outbound file artifacts,
  magic byte signature inspection, HTML/SVG sanitization, SSRF protection, and secret masking.
- SecurityQuarantineError: Raised when a security violation or malicious payload is intercepted.
"""

import ipaddress
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

# =====================================================================
# Security Exceptions
# =====================================================================


class SecurityQuarantineError(Exception):
    """Raised when an artifact or action violates security policies."""

    def __init__(self, message: str, violation_type: str = "quarantine_violation"):
        super().__init__(message)
        self.violation_type = violation_type


# =====================================================================
# Whitelists & Signatures
# =====================================================================

# Whitelisted business file extensions
ALLOWED_EXTENSIONS: Set[str] = {
    # Documents & Data
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".parquet",
    ".arrow",
    ".sql",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".py",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".svg",
    ".gif",
    ".bmp",
    ".ico",
}

# Strictly prohibited dangerous extensions (immediate rejection)
PROHIBITED_EXTENSIONS: Set[str] = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".sh",
    ".bash",
    ".zsh",
    ".bat",
    ".cmd",
    ".vbs",
    ".ps1",
    ".app",
    ".dmg",
    ".jar",
    ".apk",
    ".iso",
    ".msi",
    ".pyc",
    ".pyd",
    ".xlsm",
    ".docm",
    ".pptm",
    ".scr",
    ".pif",
    ".cpl",
    ".hta",
}

# Magic byte signatures
MAGIC_SIGNATURES: Dict[str, bytes] = {
    "png": b"\x89PNG\r\n\x1a\n",
    "pdf": b"%PDF-",
    "gif87": b"GIF87a",
    "gif89": b"GIF89a",
    "jpeg": b"\xff\xd8\xff",
    "zip": b"PK\x03\x04",
    "zip_empty": b"PK\x05\x06",
}

# Dangerous binary executable headers
BLOCKED_BINARY_HEADERS: List[Tuple[str, bytes]] = [
    ("Windows PE Executable (MZ)", b"MZ"),
    ("Linux ELF Executable", b"\x7fELF"),
    ("Mach-O 32-bit", b"\xfe\xed\xfa\xce"),
    ("Mach-O 64-bit", b"\xfe\xed\xfa\xcf"),
    ("Mach-O Universal / Java Class", b"\xca\xfe\xba\xbe"),
    ("Mach-O Reverse 32-bit", b"\xce\xfa\xed\xfe"),
    ("Mach-O Reverse 64-bit", b"\xcf\xfa\xed\xfe"),
]

# Sensitive secret regex patterns for redaction
SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Google API Key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("OpenAI API Key", re.compile(r"sk-(?:proj-|live-)?[a-zA-Z0-9_\-]{20,}")),
    ("Anthropic API Key", re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}")),
    ("E2B API Key", re.compile(r"e2b_[a-zA-Z0-9_\-]{20,}")),
    ("Bearer Token", re.compile(r"(?i)Bearer\s+([a-zA-Z0-9\-_.~+/]+=*)")),
    (
        "Generic Secret/Password in URL",
        re.compile(r"(?i)(://[^:\s]+:)([^@\s]+)(@)"),
    ),
    (
        "MongoDB Connection String",
        re.compile(r"(mongodb(?:\+srv)?://[^:\s]+:)([^@\s]+)(@)"),
    ),
]


class SecurityGuard:
    """Enterprise security perimeter enforcer and artifact quarantine engine."""

    # =================================================================
    # 1. File Artifact & Magic Byte Quarantine
    # =================================================================

    @staticmethod
    def validate_file_content(
        filename: str,
        content: Union[str, bytes],
        enforce_whitelist: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """Inspects file extension, magic byte header, and active content.

        Returns:
            (is_safe, error_message)
        """
        raw_name = Path(filename).name
        ext = Path(filename).suffix.lower()

        # 1. Check prohibited extensions
        if ext in PROHIBITED_EXTENSIONS:
            return (
                False,
                f"Security Quarantine: File '{raw_name}' has prohibited executable/macro extension '{ext}'.",
            )

        # 2. Check allowed extension whitelist (if enforced)
        if enforce_whitelist and ext and ext not in ALLOWED_EXTENSIONS:
            return (
                False,
                f"Security Quarantine: File extension '{ext}' for '{raw_name}' is not in approved whitelist.",
            )

        content_bytes = (
            content.encode("utf-8") if isinstance(content, str) else content
        )

        # 3. Check for blocked binary headers regardless of filename
        for desc, header in BLOCKED_BINARY_HEADERS:
            if content_bytes.startswith(header):
                return (
                    False,
                    f"Security Quarantine: File '{raw_name}' contains blocked binary payload signature ({desc}).",
                )

        # 4. Check specific format magic bytes if extension claims a known format
        if ext == ".png" and len(content_bytes) >= 8:
            if not content_bytes.startswith(MAGIC_SIGNATURES["png"]):
                return (
                    False,
                    f"Security Quarantine: File '{raw_name}' is not a valid PNG image (header mismatch).",
                )

        if ext == ".pdf" and len(content_bytes) >= 5:
            if not content_bytes.startswith(MAGIC_SIGNATURES["pdf"]):
                return (
                    False,
                    f"Security Quarantine: File '{raw_name}' is not a valid PDF document (header mismatch).",
                )

        if ext in (".jpg", ".jpeg") and len(content_bytes) >= 3:
            if not content_bytes.startswith(MAGIC_SIGNATURES["jpeg"]):
                return (
                    False,
                    f"Security Quarantine: File '{raw_name}' is not a valid JPEG image (header mismatch).",
                )

        # 5. Check office macro disguised payload
        if ext in (".docx", ".xlsx", ".pptx") and len(content_bytes) >= 4:
            if not (
                content_bytes.startswith(MAGIC_SIGNATURES["zip"])
                or content_bytes.startswith(MAGIC_SIGNATURES["zip_empty"])
            ):
                return (
                    False,
                    f"Security Quarantine: Office document '{raw_name}' is not a valid OpenXML package.",
                )

        return True, None

    @staticmethod
    def assert_safe_file(
        filename: str, content: Union[str, bytes], enforce_whitelist: bool = True
    ) -> None:
        """Asserts that a file passes quarantine inspection; raises SecurityQuarantineError on failure."""
        is_safe, err_msg = SecurityGuard.validate_file_content(
            filename=filename,
            content=content,
            enforce_whitelist=enforce_whitelist,
        )
        if not is_safe:
            raise SecurityQuarantineError(
                err_msg or f"File '{filename}' rejected by security policy."
            )

    # =================================================================
    # 2. Active Script & Content Sanitization (HTML / SVG)
    # =================================================================

    @staticmethod
    def sanitize_html(html_str: str) -> str:
        """Neutralizes executable scripts, iframes, objects, and event handlers in HTML."""
        if not html_str:
            return ""

        # Remove <script>...</script> tags and content
        cleaned = re.sub(
            r"(?is)<script\b[^>]*>.*?</script>", "", html_str
        )
        # Remove <iframe...>, <object...>, <embed...>
        cleaned = re.sub(r"(?is)<iframe\b[^>]*>.*?</iframe>", "", cleaned)
        cleaned = re.sub(r"(?is)<iframe\b[^>]*\/?>", "", cleaned)
        cleaned = re.sub(r"(?is)<object\b[^>]*>.*?</object>", "", cleaned)
        cleaned = re.sub(r"(?is)<embed\b[^>]*\/?>", "", cleaned)

        # Remove inline event handlers (e.g. onload=, onclick=, onerror=)
        cleaned = re.sub(
            r"""(?i)\s+on[a-z]+\s*=\s*(?:'[^']*'|"[^"]*"|[^\s>]+)""",
            "",
            cleaned,
        )

        # Neutralize javascript: and vbscript: URIs in href / src
        cleaned = re.sub(
            r"""(?i)(href|src)\s*=\s*['"]\s*(?:javascript|vbscript|data:text/html):[^'"]*['"]""",
            r'\1="#"',
            cleaned,
        )

        return cleaned

    @staticmethod
    def sanitize_svg(svg_str: str) -> str:
        """Neutralizes embedded scripts and foreign objects in SVG graphics."""
        if not svg_str:
            return ""

        # Remove <script> tags
        cleaned = re.sub(
            r"(?is)<script\b[^>]*>.*?</script>", "", svg_str
        )
        # Remove <foreignObject>
        cleaned = re.sub(
            r"(?is)<foreignObject\b[^>]*>.*?</foreignObject>", "", cleaned
        )

        # Remove inline event handlers
        cleaned = re.sub(
            r"""(?i)\s+on[a-z]+\s*=\s*(?:'[^']*'|"[^"]*"|[^\s>]+)""",
            "",
            cleaned,
        )

        # Neutralize xlink:href or href with javascript
        cleaned = re.sub(
            r"""(?i)(?:xlink:)?href\s*=\s*['"]\s*javascript:[^'"]*['"]""",
            'href="#"',
            cleaned,
        )

        return cleaned

    # =================================================================
    # 3. SSRF & Inbound Network Guard
    # =================================================================

    @staticmethod
    def is_safe_url(target_url: str) -> Tuple[bool, Optional[str]]:
        """Validates that a URL does not target loopback, RFC1918 private subnets, or cloud metadata."""
        if not target_url or not target_url.strip():
            return False, "Empty URL"

        try:
            parsed = urlparse(target_url.strip())
        except Exception as e:
            return False, f"Invalid URL structure: {e}"

        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            return False, f"Unsupported URL scheme '{scheme}'. Only http and https are allowed."

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False, "Missing hostname in URL"

        # Block cloud metadata hostnames
        if hostname in ("metadata.google.internal", "instance-data", "169.254.169.254"):
            return False, f"SSRF Block: Access to cloud metadata endpoint ({hostname}) is forbidden."

        # Check IP address targets
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_loopback:
                return (
                    False,
                    f"SSRF Block: Access to loopback address ({hostname}) is forbidden.",
                )
            if ip.is_private:
                return (
                    False,
                    f"SSRF Block: Access to private network address ({hostname}) is forbidden.",
                )
            if ip.is_link_local:
                return (
                    False,
                    f"SSRF Block: Access to link-local address ({hostname}) is forbidden.",
                )
        except ValueError:
            # Hostname is a domain name (not an IP address), allowed
            pass

        return True, None

    # =================================================================
    # 4. Sensitive Secret Redaction
    # =================================================================

    @staticmethod
    def redact_secrets(text: str) -> str:
        """Masks API keys, bearer tokens, and connection credentials from logs and outputs."""
        if not text:
            return ""

        redacted = text
        for name, pattern in SECRET_PATTERNS:
            if "URL" in name or "MongoDB" in name:
                redacted = pattern.sub(r"\1***:***\3", redacted)
            elif name == "Bearer Token":
                redacted = pattern.sub(r"Bearer [REDACTED_TOKEN]", redacted)
            else:
                redacted = pattern.sub(lambda m: m.group(0)[:6] + "..." + "[REDACTED]", redacted)

        return redacted
