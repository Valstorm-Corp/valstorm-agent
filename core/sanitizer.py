"""Local secret redaction, PCI payment card guard, and LLM egress sanitizer.

Prevents sensitive API keys, auth tokens, database credentials, Azure connection strings,
environment secrets, private keys, and PCI-DSS payment cardholder data (Luhn-validated PANs,
CVV/CVC, card expiry) from leaking to cloud LLM providers or persisted chat logs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

# Sensitive environment key fragments to discover and mask
SENSITIVE_ENV_SUBSTRINGS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "DATABASE_URL",
    "POSTGRES_URI",
    "POSTGRES_URL",
    "MONGO_URI",
    "MONGO_URL",
    "REDIS_URL",
    "REDIS_URI",
    "CONNECTION_STRING",
    "CONN_STR",
    "CONNSTRING",
    "DSN",
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
    # Generic .env line key-value matching (UPPERCASE env var names to avoid matching Python/JS code vars)
    (
        re.compile(
            r"""(?m)^([ \t]*(?:export[ \t]+)?([A-Z0-9_]*?(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|AUTH|DATABASE_URL|POSTGRES_URI|POSTGRES_URL|MONGO_URI|MONGO_URL|REDIS_URL|REDIS_URI|CONNECTION_STRING|CONN_STR|CONNSTRING|DSN|PRIVATE|CREDENTIAL|WEBHOOK|BEARER|SALT|ENCRYPT|SIGNING)[A-Z0-9_]*?)[ \t]*=[ \t]*)(['"]?)([^\r\n"'\s<>{}$]{6,})\3"""
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
    # E2B API Key
    (
        re.compile(r"\be2b_[a-zA-Z0-9_\-]{20,}\b"),
        "<REDACTED_E2B_KEY>",
    ),
    # Database URIs with credentials (including password-only redis://:pass@host and SQLAlchemy drivers)
    (
        re.compile(
            r"""(\b(?:mongodb(?:\+srv)?|postgres(?:ql)?(?:\+[a-z0-9_]+)?|mysql(?:\+[a-z0-9_]+)?|rediss?|amqps?|mssql(?:\+[a-z0-9_]+)?|sqlserver):\/\/[^:\/\s@"']*:)(?:<REDACTED_DB_PASSWORD>|[^@\s"'<>]+)(@)""",
            re.IGNORECASE,
        ),
        r"\1<REDACTED_DB_PASSWORD>\2",
    ),
    # Azure Storage AccountKey in connection strings
    (
        re.compile(r"""(?i)(\bAccountKey\s*=\s*)([^;\s"'<>]{16,})"""),
        r"\1<REDACTED_AZURE_KEY>",
    ),
    # Azure Communication Services (ACS) connection string accesskey
    (
        re.compile(r"""(?i)(endpoint\s*=\s*https?:\/\/[^;\s"']+;\s*accesskey\s*=\s*)([^;\s"'<>]{16,})"""),
        r"\1<REDACTED_AZURE_KEY>",
    ),
    (
        re.compile(r"""(?i)(;\s*accesskey\s*=\s*)([^;\s"'<>]{16,})"""),
        r"\1<REDACTED_AZURE_KEY>",
    ),
    # Azure Service Bus / Event Hubs SharedAccessKey
    (
        re.compile(r"""(?i)(\bSharedAccessKey\s*=\s*)([^;\s"'<>]{16,})"""),
        r"\1<REDACTED_AZURE_KEY>",
    ),
    # Azure SAS URL signature parameter
    (
        re.compile(r"""(?i)(https?:\/\/[^\s"'<>]*(?:windows\.net|azure\.com)[^\s"'<>]*[?&]sig=)([^&\s"'<>]{16,})"""),
        r"\1<REDACTED_AZURE_KEY>",
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
    (
        re.compile(r"(?i)\b(Bearer\s+)([a-zA-Z0-9_\-\.]{20,})\b"),
        r"\1<REDACTED_BEARER_TOKEN>",
    ),
    # HTTP API key headers (x-api-key, Ocp-Apim-Subscription-Key, api-key, x-auth-token, x-access-token)
    (
        re.compile(
            r"""(?i)(["']?(?:x-api-key|x-auth-token|x-access-token|ocp-apim-subscription-key|api-key)["']?\s*[:=]\s*["']?)([^\s"'`,;}\[\]<>]{8,})"""
        ),
        r"\1<REDACTED_API_KEY>",
    ),
    # Quoted key-value credential / password assignments
    (
        re.compile(
            r"""(?i)(["']?(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|signing[_-]?secret|webhook[_-]?secret|encryption[_-]?key|master[_-]?key|service[_-]?role[_-]?key|db[_-]?password|smtp[_-]?password|password|passwd)["']?\s*[:=]\s*["'])([^"'\r\n<>]{6,})(["'])"""
        ),
        r"\1<REDACTED_SECRET>\3",
    ),
]

_UNQUOTED_CRED_RE = re.compile(
    r"""(?i)\b((?:password|passwd|pwd|db[_-]?password|smtp[_-]?password|client[_-]?secret|api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|private[_-]?key|signing[_-]?secret|webhook[_-]?secret)\s*=\s*)([^\s&;,"'`<>()\[\]{}]{6,})"""
)

_PAN_CANDIDATE_RE = re.compile(
    r"\d{4}(?:-\d{4}){3}-\d{3}|\d{4}(?: \d{4}){3} \d{3}|\d{4}(?:-\d{4}){2}-\d{1,4}|\d{4}(?: \d{4}){2} \d{1,4}|\d{4}-\d{6}-\d{4,5}|\d{4} \d{6} \d{4,5}|\d{4}-\d{4}-\d{5}|\d{4} \d{4} \d{5}|\d{13,19}"
)

_CVV_CONTEXT_RE = re.compile(
    r"""(?i)\b((?:card[ _-]?)?(?:cvv2?|cvc2?|cvn|security[ _-]?code|card[ _-]?verification[ _-]?(?:value|code|number))["']?(?:\s+(?:code|number|is))?\s*[:=]?\s*["']?)(\d{3,4})\b"""
)

_EXPIRY_CONTEXT_RE = re.compile(
    r"""(?i)\b((?:card[ _-]?)?(?:exp(?:iry|iration)?(?:[ _-]?date)?|valid[ _-]?thru|good[ _-]?thru|expires)["']?(?:\s+(?:date|on|at|is))?\s*[:=]?\s*["']?)((?:0[1-9]|1[0-2])\s*[/-]\s*(?:20\d{2}|\d{2}))\b"""
)

_EXPIRY_MONTH_RE = re.compile(
    r"""(?i)\b((?:card[ _-]?)?exp(?:iry|iration)?[ _-]?(?:month|mm)["']?\s*[:=]\s*["']?)(0?[1-9]|1[0-2])\b"""
)

_EXPIRY_YEAR_RE = re.compile(
    r"""(?i)\b((?:card[ _-]?)?exp(?:iry|iration)?[ _-]?(?:year|yy|yyyy)["']?\s*[:=]\s*["']?)(20\d{2}|\d{2})\b"""
)

_PAN_ADJACENT_EXP_CVV_RE = re.compile(
    r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t/ ][ \t]*)(?:0[1-9]|1[0-2])[/-](?:20\d{2}|\d{2})([ \t]*[,|\t/ ][ \t]*)\d{3,4}\b"
)
_PAN_ADJACENT_CVV_EXP_RE = re.compile(
    r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t/ ][ \t]*)\d{3,4}([ \t]*[,|\t/ ][ \t]*)(?:0[1-9]|1[0-2])[/-](?:20\d{2}|\d{2})\b"
)
_PAN_ADJACENT_EXP_ONLY_RE = re.compile(
    r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t/ ][ \t]*)(?:0[1-9]|1[0-2])[/-](?:20\d{2}|\d{2})\b"
)
_PAN_ADJACENT_CVV_ONLY_RE = re.compile(
    r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t][ \t]*)\d{3,4}\b"
)

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
    "string",
    "optional",
    "boolean",
    "number",
    "required",
    "masked",
    "redacted",
}

_CODE_EXPR_PREFIXES = (
    "self.",
    "this.",
    "os.",
    "env.",
    "process.",
    "settings.",
    "config.",
    "props.",
    "args.",
    "req.",
    "ctx.",
    "params.",
    "data.",
    "user.",
    "payload.",
    "body.",
    "headers.",
)

# Explicit dictionary keys masked when sanitizing structured dicts (e.g. tool_calls, execution_trace)
_SENSITIVE_STRUCT_CVV_KEYS = frozenset({
    "cvv",
    "cvv2",
    "cvc",
    "cvc2",
    "cvn",
    "security_code",
    "card_security_code",
})

_SENSITIVE_STRUCT_EXPIRY_KEYS = frozenset({
    "exp_date",
    "expiry_date",
    "expiration_date",
    "card_exp",
    "card_expiry",
    "exp_month",
    "exp_year",
    "card_exp_month",
    "card_exp_year",
})

_SENSITIVE_STRUCT_SECRET_KEYS = frozenset({
    "password",
    "passwd",
    "db_password",
    "smtp_password",
    "client_secret",
    "api_key",
    "api_secret",
    "secret_key",
    "access_token",
    "refresh_token",
    "auth_token",
    "private_key",
    "signing_secret",
    "webhook_secret",
    "encryption_key",
    "x-api-key",
    "x_api_key",
    "ocp-apim-subscription-key",
    "connection_string",
    "azure_acs_email_connection_string",
})


def _is_code_expression_value(key_part: str, val: str) -> bool:
    """Returns True when an unquoted key=value match is clearly a code expression or placeholder."""
    lower_val = val.lower()
    if lower_val in IGNORED_ENV_VALUES:
        return True
    if any(lower_val.startswith(prefix) for prefix in _CODE_EXPR_PREFIXES):
        return True
    if val.startswith("$"):
        rest = val[1:]
        if rest and all(ch.isupper() or ch.isdigit() or ch == "_" for ch in rest):
            return True
    key_name = key_part.rstrip("= \t").lower()
    if lower_val == key_name:
        return True
    return False


def _passes_luhn(digits: str) -> bool:
    """Standard ISO/IEC 7812 Luhn checksum validation."""
    total = 0
    double = False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return (total % 10) == 0


def _is_valid_card_iin_and_length(digits: str) -> bool:
    """Validates known PCI card network IIN/BIN prefixes and allowed PAN lengths."""
    length = len(digits)
    if length < 13 or length > 19:
        return False
    p1 = int(digits[0])
    p2 = int(digits[:2])
    p3 = int(digits[:3])
    p4 = int(digits[:4])
    p6 = int(digits[:6])

    # Visa: 4, length 13, 16, 19
    if p1 == 4 and length in (13, 16, 19):
        return True
    # Mastercard: 51-55 or 2221-2720, length 16
    if length == 16 and ((51 <= p2 <= 55) or (2221 <= p4 <= 2720)):
        return True
    # American Express: 34 or 37, length 15
    if length == 15 and p2 in (34, 37):
        return True
    # Discover: 6011, 65, 644-649, 622126-622925, length 16-19
    if 16 <= length <= 19 and (
        p4 == 6011 or p2 == 65 or (644 <= p3 <= 649) or (622126 <= p6 <= 622925)
    ):
        return True
    # Diners Club: 300-305, 36, 38-39, length 14-19
    if 14 <= length <= 19 and ((300 <= p3 <= 305) or p2 in (36, 38, 39)):
        return True
    # JCB: 3528-3589, length 16-19
    if 16 <= length <= 19 and (3528 <= p4 <= 3589):
        return True
    # UnionPay: 62, length 16-19
    if 16 <= length <= 19 and p2 == 62:
        return True
    # Maestro: 50, 56-58, 6304, 6759, 6761-6763, length 13-19
    if 13 <= length <= 19 and (
        p2 == 50 or (56 <= p2 <= 58) or p4 in (6304, 6759) or (6761 <= p4 <= 6763)
    ):
        return True
    return False


def _has_valid_pan_left_boundary(text: str, start: int) -> bool:
    if start == 0:
        return True
    prev = text[start - 1]
    if prev.isascii() and prev.isalnum():
        return False
    if prev in ("_", "+", "."):
        return False
    if prev == "-" and start >= 2:
        before_dash = text[start - 2]
        if (before_dash.isascii() and before_dash.isalnum()) or before_dash == "_":
            return False
    return True


def _has_valid_pan_right_boundary(text: str, end: int) -> bool:
    if end >= len(text):
        return True
    nxt = text[end]
    if nxt.isascii() and nxt.isalnum():
        return False
    if nxt == "_":
        return False
    if nxt == "." and end + 1 < len(text) and text[end + 1].isascii() and text[end + 1].isdigit():
        return False
    if nxt == "-" and end + 1 < len(text) and text[end + 1].isascii() and text[end + 1].isalnum():
        return False
    return True


def _is_valid_pan_slice(slice_text: str) -> bool:
    digits = "".join(ch for ch in slice_text if "0" <= ch <= "9")
    return _is_valid_card_iin_and_length(digits) and _passes_luhn(digits)


def redact_payment_cards(text: str) -> str:
    """Detects and redacts Luhn + IIN/BIN validated PCI payment card PANs."""
    if not text or not any("0" <= ch <= "9" for ch in text):
        return text

    out_parts: List[str] = []
    last_end = 0
    search_pos = 0
    text_len = len(text)

    while search_pos < text_len:
        m = _PAN_CANDIDATE_RE.search(text, search_pos)
        if m is None:
            break
        start, end = m.start(), m.end()
        matched = m.group(0)

        if _has_valid_pan_left_boundary(text, start):
            if _has_valid_pan_right_boundary(text, end) and _is_valid_pan_slice(matched):
                out_parts.append(text[last_end:start])
                out_parts.append("<REDACTED_PAYMENT_CARD>")
                last_end = end
                search_pos = end
                continue
            # Fallback for 16-digit space-separated PAN followed by a 3-digit CVV or 4-digit year
            if (
                len(matched) == 23
                and matched[4] == " "
                and matched[9] == " "
                and matched[14] == " "
                and matched[19] == " "
            ):
                prefix_end = start + 19
                if _has_valid_pan_right_boundary(text, prefix_end) and _is_valid_pan_slice(
                    text[start:prefix_end]
                ):
                    out_parts.append(text[last_end:start])
                    out_parts.append("<REDACTED_PAYMENT_CARD>")
                    last_end = prefix_end
                    search_pos = prefix_end
                    continue

        search_pos = start + 1

    if last_end == 0:
        return text
    out_parts.append(text[last_end:])
    return "".join(out_parts)


def _redact_unquoted_credentials(text: str) -> str:
    if not _UNQUOTED_CRED_RE.search(text):
        return text

    def _repl(m: re.Match[str]) -> str:
        prefix, val = m.group(1), m.group(2)
        if _is_code_expression_value(prefix, val):
            return m.group(0)
        return f"{prefix}<REDACTED_SECRET>"

    return _UNQUOTED_CRED_RE.sub(_repl, text)


def redact_cvv_and_expiry(text: str) -> str:
    """Redacts contextual CVV/CVC/security_code and card expiry values (plus PAN-adjacent tabular rows)."""
    current = text

    # Contextual CVV/CVC/security_code (avoid matching phone fragments like `cvv 123-4567`)
    if _CVV_CONTEXT_RE.search(current):
        out_parts: List[str] = []
        last_end = 0
        for m in _CVV_CONTEXT_RE.finditer(current):
            end = m.end()
            followed_by_dash_or_dot_digit = (
                end + 1 < len(current)
                and current[end] in ("-", ".")
                and current[end + 1].isascii()
                and current[end + 1].isdigit()
            )
            if not followed_by_dash_or_dot_digit:
                out_parts.append(current[last_end : m.start()])
                out_parts.append(f"{m.group(1)}<REDACTED_CVV>")
                last_end = end
        if last_end > 0:
            out_parts.append(current[last_end:])
            current = "".join(out_parts)

    # Contextual Card Expiry MM/YY or MM/YYYY (avoid truncating full dates like `05/15/2027`)
    if _EXPIRY_CONTEXT_RE.search(current):
        out_parts = []
        last_end = 0
        for m in _EXPIRY_CONTEXT_RE.finditer(current):
            end = m.end()
            followed_by_date_part = (
                end + 1 < len(current)
                and current[end] in ("/", "-")
                and current[end + 1].isascii()
                and current[end + 1].isdigit()
            )
            if not followed_by_date_part:
                out_parts.append(current[last_end : m.start()])
                out_parts.append(f"{m.group(1)}<REDACTED_CARD_EXPIRY>")
                last_end = end
        if last_end > 0:
            out_parts.append(current[last_end:])
            current = "".join(out_parts)

    current = _EXPIRY_MONTH_RE.sub(r"\1<REDACTED_CARD_EXPIRY>", current)
    current = _EXPIRY_YEAR_RE.sub(r"\1<REDACTED_CARD_EXPIRY>", current)

    # PAN-adjacent tabular/CSV/pipe/space expiry & CVV
    if "<REDACTED_PAYMENT_CARD>" in current:
        current = _PAN_ADJACENT_EXP_CVV_RE.sub(r"\1<REDACTED_CARD_EXPIRY>\2<REDACTED_CVV>", current)
        current = _PAN_ADJACENT_CVV_EXP_RE.sub(r"\1<REDACTED_CVV>\2<REDACTED_CARD_EXPIRY>", current)
        current = _PAN_ADJACENT_EXP_ONLY_RE.sub(r"\1<REDACTED_CARD_EXPIRY>", current)
        current = _PAN_ADJACENT_CVV_ONLY_RE.sub(r"\1<REDACTED_CVV>", current)

    return current


class SecretSanitizer:
    """High-performance in-memory sanitizer matching known secret signatures, PCI PAN/CVV, and loaded env variables."""

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
        """Sanitizes text, masking API keys, credentials, PCI payment cards/CVVs, and environment secrets.

        Executes in sub-millisecond time.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return ""

        result = text

        # 1. Exact Environment / Custom Secret Masking (runs first so full env secrets keep their env label)
        for secret_val, replacement_label in self._exact_replacements:
            if secret_val in result:
                result = result.replace(secret_val, replacement_label)

        # 2. Fast Native Rust Regex + Luhn Sanitize pass (if compiled accelerator is available)
        try:
            from core.native_bridge import native_sanitize_text

            native_res = native_sanitize_text(result)
            if native_res is not None:
                result = native_res
        except Exception:
            pass

        # 3. Known API / Token / Connection String Pattern Masking
        for pattern, replacement in SECRET_PATTERNS:
            result = pattern.sub(replacement, result)

        # 4. Smart unquoted credential key=value masking
        result = _redact_unquoted_credentials(result)

        # 5. Luhn + IIN validated PCI Payment Card PAN masking
        result = redact_payment_cards(result)

        # 6. Contextual CVV/CVC, Card Expiry, and PAN-adjacent CSV/table masking
        result = redact_cvv_and_expiry(result)

        return result

    def sanitize_structure(self, data: Any) -> Any:
        """Recursively sanitizes sensitive strings and PCI/credential dict fields while preserving binary bytes."""
        if data is None or isinstance(data, (bytes, bytearray, bool)):
            return data
        if isinstance(data, str):
            return self.sanitize(data)
        if isinstance(data, int):
            s = str(data)
            if 13 <= len(s) <= 19 and _is_valid_pan_slice(s):
                return "<REDACTED_PAYMENT_CARD>"
            return data
        if isinstance(data, dict):
            clean_dict: Dict[Any, Any] = {}
            for k, v in data.items():
                k_lower = str(k).strip().lower()
                if isinstance(v, (dict, list, tuple, set)):
                    clean_dict[k] = self.sanitize_structure(v)
                elif isinstance(v, (bytes, bytearray, bool)) or v is None:
                    clean_dict[k] = v
                elif k_lower in _SENSITIVE_STRUCT_CVV_KEYS:
                    v_str = str(v).strip()
                    if re.fullmatch(r"\d{3,4}", v_str):
                        clean_dict[k] = "<REDACTED_CVV>"
                    else:
                        clean_dict[k] = self.sanitize_structure(v)
                elif k_lower in _SENSITIVE_STRUCT_EXPIRY_KEYS:
                    v_str = str(v).strip()
                    if v_str and not v_str.startswith("<REDACTED_"):
                        clean_dict[k] = "<REDACTED_CARD_EXPIRY>"
                    else:
                        clean_dict[k] = v
                elif k_lower in _SENSITIVE_STRUCT_SECRET_KEYS:
                    v_str = str(v).strip()
                    if len(v_str) >= 6 and v_str.lower() not in IGNORED_ENV_VALUES and not v_str.startswith("<REDACTED_"):
                        clean_dict[k] = "<REDACTED_SECRET>"
                    else:
                        clean_dict[k] = self.sanitize_structure(v)
                else:
                    clean_dict[k] = self.sanitize_structure(v)
            return clean_dict
        if isinstance(data, list):
            return [self.sanitize_structure(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.sanitize_structure(item) for item in data)
        if isinstance(data, set):
            return [self.sanitize_structure(item) for item in data]
        return data


# Global singleton instance
_GLOBAL_SANITIZER = SecretSanitizer()


def sanitize_text(text: Any) -> str:
    """Convenience helper to sanitize text using the global SecretSanitizer."""
    return _GLOBAL_SANITIZER.sanitize(text)


def sanitize_structure(data: Any) -> Any:
    """Convenience helper to recursively sanitize dicts/lists/nested payloads using the global SecretSanitizer."""
    return _GLOBAL_SANITIZER.sanitize_structure(data)


def sanitize_messages(messages: List[Any]) -> List[Any]:
    """Defense-in-depth egress sanitizer for Message lists before dispatching to any LLM provider.

    Returns sanitized shallow copies of Message objects (preserving binary thought_signature bytes
    and avoiding in-place mutation surprises).
    """
    sanitized_list: List[Any] = []
    for msg in messages:
        if msg is None:
            continue
        try:
            updates: Dict[str, Any] = {}
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                clean_content = sanitize_text(content)
                if clean_content != content:
                    updates["content"] = clean_content
            body = getattr(msg, "body", None)
            if isinstance(body, str) and body:
                clean_body = sanitize_text(body)
                if clean_body != body:
                    updates["body"] = clean_body
            sys_prompt = getattr(msg, "system_prompt", None)
            if isinstance(sys_prompt, str) and sys_prompt:
                clean_sys = sanitize_text(sys_prompt)
                if clean_sys != sys_prompt:
                    updates["system_prompt"] = clean_sys

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                new_tcs = []
                tc_changed = False
                for tc in tool_calls:
                    args = getattr(tc, "arguments", None)
                    if args is not None:
                        clean_args = sanitize_structure(args)
                        if clean_args != args:
                            new_tcs.append(tc.model_copy(update={"arguments": clean_args}))
                            tc_changed = True
                            continue
                    new_tcs.append(tc)
                if tc_changed:
                    updates["tool_calls"] = new_tcs

            tool_result = getattr(msg, "tool_result", None)
            if tool_result is not None:
                tr_out = getattr(tool_result, "output", None)
                if tr_out is not None:
                    clean_tr_out = sanitize_text(tr_out) if isinstance(tr_out, str) else sanitize_structure(tr_out)
                    if clean_tr_out != tr_out:
                        updates["tool_result"] = tool_result.model_copy(update={"output": clean_tr_out})

            if updates and hasattr(msg, "model_copy"):
                sanitized_list.append(msg.model_copy(update=updates))
            else:
                sanitized_list.append(msg)
        except Exception:
            sanitized_list.append(msg)
    return sanitized_list


def register_secret_to_redact(name: str, value: str) -> None:
    """Registers a secret to be masked globally."""
    _GLOBAL_SANITIZER.add_secret(name, value)
