"""Unit tests for local secret redaction and egress sanitizer."""

import os
import pytest

from core.sanitizer import SecretSanitizer, sanitize_text, register_secret_to_redact
from core.tools import ToolRegistry, tool


class TestSecretSanitizerPatterns:
    """Tests regex pattern matching for well-known API keys and credentials."""

    def test_openai_and_project_keys_redacted(self):
        k1 = "sk-proj-" + "abcdef123456789012345678"
        k2 = "sk-" + "12345678901234567890"
        text = f"OpenAI key: {k1} and legacy {k2}"
        sanitized = sanitize_text(text)
        assert ("sk-proj-" + "abcdef") not in sanitized
        assert ("sk-" + "1234567890") not in sanitized
        assert "<REDACTED_OPENAI_KEY>" in sanitized

    def test_anthropic_key_redacted(self):
        k = "sk-ant-" + "api03-abcdef1234567890123456"
        text = f"Anthropic key is {k}"
        sanitized = sanitize_text(text)
        assert ("sk-ant-" + "api03") not in sanitized
        assert "<REDACTED_ANTHROPIC_KEY>" in sanitized

    def test_google_cloud_gemini_key_redacted(self):
        k = "AIzaSy" + "abcdef123456789012345678901234567"
        text = f"Gemini Key: {k}"
        sanitized = sanitize_text(text)
        assert ("AIza" + "Sy") not in sanitized
        assert "<REDACTED_GOOGLE_API_KEY>" in sanitized

    def test_github_tokens_redacted(self):
        pat = "ghp_" + "1234567890abcdefghijklmnopqrstuvwxyz"
        pat_fine = "github_pat_" + "11ABCD_1234567890abcdefghijklmnopqrstuvwxyz1234567890abcdefghij"
        text = f"User PAT: {pat}, Fine-grained: {pat_fine}"
        sanitized = sanitize_text(text)
        assert pat not in sanitized
        assert pat_fine not in sanitized
        assert "<REDACTED_GITHUB_PAT>" in sanitized
        assert "<REDACTED_GITHUB_FINE_GRAINED_PAT>" in sanitized

    def test_slack_token_and_webhook_redacted(self):
        token = "xoxb-" + "1234567890-123456789012-abcdefghijklmnopqrstuvwx"
        webhook = "https://" + "hooks.slack.com/services/T" + "00000000/B" + "00000000/mockhooksecret"
        text = f"Slack auth: {token}, Webhook: {webhook}"
        sanitized = sanitize_text(text)
        assert token not in sanitized
        assert webhook not in sanitized
        assert "<REDACTED_SLACK_TOKEN>" in sanitized
        assert "<REDACTED_SLACK_WEBHOOK>" in sanitized

    def test_aws_access_key_redacted(self):
        k = "AKIA" + "IOSFODNN7EXAMPLE"
        text = f"AWS Access: {k}"
        sanitized = sanitize_text(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in sanitized
        assert "<REDACTED_AWS_ACCESS_KEY_ID>" in sanitized

    def test_stripe_keys_redacted(self):
        k = "sk_live_" + "51Abcdefghijklmnopqrstuvwxyz12345"
        text = f"Stripe live key: {k}"
        sanitized = sanitize_text(text)
        assert "sk_live_" not in sanitized
        assert "<REDACTED_STRIPE_KEY>" in sanitized

    def test_database_uri_passwords_redacted(self):
        mongo = "mongodb+srv://admin:SuperSecretPass123!@cluster0.abcde.mongodb.net/prod"
        postgres = "postgresql://postgres:MyRootDbPassword999@10.0.0.1:5432/valstorm"
        redis = "redis://default:RedisPassWordSecret!@redis-host:6379/0"
        
        text = f"Connections:\n{mongo}\n{postgres}\n{redis}"
        sanitized = sanitize_text(text)
        
        assert "SuperSecretPass123!" not in sanitized
        assert "MyRootDbPassword999" not in sanitized
        assert "RedisPassWordSecret!" not in sanitized
        assert "mongodb+srv://admin:<REDACTED_DB_PASSWORD>@cluster0.abcde.mongodb.net/prod" in sanitized
        assert "postgresql://postgres:<REDACTED_DB_PASSWORD>@10.0.0.1:5432/valstorm" in sanitized
        assert "redis://default:<REDACTED_DB_PASSWORD>@redis-host:6379/0" in sanitized

    def test_private_key_blocks_redacted(self):
        rsa_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y1W9ZqL...dummy...base64\n"
            "-----END RSA PRIVATE KEY-----"
        )
        text = f"Config with RSA certificate:\n{rsa_key}\nHost: example.com"
        sanitized = sanitize_text(text)
        assert "MIIEowIBAAKCAQEA0Y1W9ZqL" not in sanitized
        assert "<REDACTED_PRIVATE_KEY_BLOCK>" in sanitized
        assert "Host: example.com" in sanitized

    def test_jwt_token_redacted(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Bearer token: {jwt}"
        sanitized = sanitize_text(text)
        assert jwt not in sanitized
        assert "<REDACTED_JWT_TOKEN>" in sanitized


class TestEnvironmentSecretMasking:
    """Tests dynamic discovery and masking of loaded os.environ values."""

    def test_env_secrets_automatically_masked(self, monkeypatch):
        monkeypatch.setenv("SENDGRID_API_KEY", "SG.my_secret_sendgrid_key_9999888877776666")
        monkeypatch.setenv("INTERNAL_AUTH_SECRET", "super_duper_internal_master_token_2026")
        
        sanitizer = SecretSanitizer(include_env=True)
        text = "Dump config: key is SG.my_secret_sendgrid_key_9999888877776666 and token is super_duper_internal_master_token_2026"
        out = sanitizer.sanitize(text)
        
        assert "super_duper_internal_master_token_2026" not in out
        assert "<REDACTED_ENV_SECRET:INTERNAL_AUTH_SECRET>" in out

    def test_custom_registered_secrets(self):
        sanitizer = SecretSanitizer(custom_secrets={"MY_OAUTH_CLIENT_SECRET": "xyz_client_secret_value_12345"})
        text = "Client secret: xyz_client_secret_value_12345"
        out = sanitizer.sanitize(text)
        assert "xyz_client_secret_value_12345" not in out
        assert "<REDACTED_CUSTOM_SECRET:MY_OAUTH_CLIENT_SECRET>" in out


class TestToolRegistryIntegration:
    """Tests that ToolRegistry automatically sanitizes tool execution outputs."""

    def test_tool_output_is_sanitized_on_execution(self):
        registry = ToolRegistry()

        @registry.register
        def fetch_cloud_config() -> str:
            """Fetch fake config with sensitive credentials."""
            return "DATABASE_URL=postgresql://valstorm_user:SuperMegaPassword@localhost:5432/db"

        result = registry.execute("fetch_cloud_config")
        assert result.is_error is False
        assert "SuperMegaPassword" not in result.output
        assert ("<REDACTED_DB_PASSWORD>" in result.output or "<REDACTED_ENV_SECRET:DATABASE_URL>" in result.output)
