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


class TestPCIPaymentCardAndCVVRedaction:
    """Tests Luhn + IIN validated PCI PAN redaction, CVV/CVC/expiry redaction, and zero false positives."""

    def test_visa_mastercard_amex_discover_diners_jcb_redacted(self, monkeypatch):
        # Force pure-Python path so Python implementation is tested independently of native Rust
        monkeypatch.setattr("core.native_bridge.native_sanitize_text", lambda _t: None)

        pans = [
            "4111" + "1111" + "1111" + "1111",       # Visa 16 contiguous
            "4242-" + "4242-" + "4242-" + "4242",    # Visa 16 hyphenated
            "4012 " + "8888 " + "8888 " + "1881",    # Visa 16 spaced
            "4111" + "1111" + "1111" + "9",          # Visa 13 contiguous
            "5555-" + "5555-" + "5555-" + "4444",    # Mastercard 5x
            "2221 " + "0000 " + "0000 " + "0009",    # Mastercard 2-series
            "3782" + "822463" + "10005",             # Amex 15 contiguous
            "3714-" + "496353-" + "98431",           # Amex 15 hyphenated
            "3400 " + "000000 " + "00009",           # Amex 15 spaced
            "6011-" + "1111-" + "1111-" + "1117",    # Discover 6011
            "6500 " + "0000 " + "0000 " + "0002",    # Discover 65
            "3056 " + "930902 " + "5904",            # Diners Club 14
            "3528-" + "0000-" + "0000-" + "0007",    # JCB 16
        ]

        sanitizer = SecretSanitizer(include_env=False)
        for pan in pans:
            out = sanitizer.sanitize(f"Please charge card {pan} today.")
            assert pan not in out, f"PAN {pan} was not redacted: {out}"
            assert out == "Please charge card <REDACTED_PAYMENT_CARD> today."

    def test_zero_false_positives_on_clinical_ids_phones_npis_and_code(self, monkeypatch):
        monkeypatch.setattr("core.native_bridge.native_sanitize_text", lambda _t: None)
        sanitizer = SecretSanitizer(include_env=False)

        safe_samples = [
            "Provider NPI: 1234567893 and secondary NPI 1245319599",
            "Call +1-415-555-2671 or (555) 234-5678 or +49 151 2345 6789",
            "Ticket tick_Z5NdByhscBapsiME patient pat_4111111111111111 claim clm-4111111111111111 MRN4111111111111111",
            "Non-Luhn numbers 4111-1111-1111-1112 and 5555-5555-5555-4445 and 1234-5678-9012-3456",
            "Tracking 41111111111111119999 and snowflake 17325735370001234567",
            "Created 2026-11-25T22:25:37Z stamp 20261125143059 license expires 05/15/2027",
            "Float 3.1415926535897932 and metric 4.1111111111111111",
            "UUID 550e8400-e29b-41d4-a716-446655440000 ICD-10 E11.9 CPT 99213 NDC 00093-3109-05",
            "password = self.password\nclient_secret = settings.client_secret\npassword = os.getenv('PASS')\npassword=${DB_PASSWORD}",
        ]

        for sample in safe_samples:
            assert sanitizer.sanitize(sample) == sample, f"False positive on: {sample}"

    def test_cvv_cvc_security_code_and_expiry_redacted(self, monkeypatch):
        monkeypatch.setattr("core.native_bridge.native_sanitize_text", lambda _t: None)
        sanitizer = SecretSanitizer(include_env=False)

        payload = '{"cvv": "742", "cvc": 8431, "security_code": "019", "exp_date": "09/28", "exp_month": "09", "exp_year": "2028"}'
        out = sanitizer.sanitize(payload)
        assert "742" not in out
        assert "8431" not in out
        assert "019" not in out
        assert "09/28" not in out
        assert '"cvv": "<REDACTED_CVV>"' in out
        assert '"cvc": <REDACTED_CVV>' in out
        assert '"security_code": "<REDACTED_CVV>"' in out
        assert '"exp_date": "<REDACTED_CARD_EXPIRY>"' in out
        assert '"exp_month": "<REDACTED_CARD_EXPIRY>"' in out
        assert '"exp_year": "<REDACTED_CARD_EXPIRY>"' in out

        prose = "Card Security Code: 991, CVV 312, Valid Thru: 11/2029, expires 04/27"
        out_prose = sanitizer.sanitize(prose)
        assert "991" not in out_prose
        assert "312" not in out_prose
        assert "11/2029" not in out_prose
        assert "04/27" not in out_prose
        assert "<REDACTED_CVV>" in out_prose
        assert "<REDACTED_CARD_EXPIRY>" in out_prose


class TestExpandedCredentialsAndConnectionStrings:
    """Tests expanded DB URIs, Azure connection strings, x-api-key, and password/client_secret redaction."""

    def test_expanded_connection_strings_headers_and_passwords_redacted(self, monkeypatch):
        monkeypatch.setattr("core.native_bridge.native_sanitize_text", lambda _t: None)
        sanitizer = SecretSanitizer(include_env=False)

        samples = {
            "PgSecret99!": "postgres://admin:" + "PgSecret99!" + "@db.internal:5432/valstorm",
            "AsyncPass123": "postgresql+asyncpg://app:" + "AsyncPass123" + "@pg.local:5432/app",
            "RedisSecret88!": "redis://default:" + "RedisSecret88!" + "@redis.local:6379/0",
            "AzureRedisKey999=": "rediss://:" + "AzureRedisKey999=" + "@cache.redis.cache.windows.net:6380/0",
            "MongoPass777!": "mongodb+srv://root:" + "MongoPass777!" + "@cluster0.mongodb.net/prod",
            "MockAzureStorageKeyBase64Value1234567890==": (
                "DefaultEndpointsProtocol=https;AccountName=valstormprod;AccountKey="
                + "MockAzureStorageKeyBase64Value1234567890=="
                + ";EndpointSuffix=core.windows.net"
            ),
            "MockAcsAccessKeyBase64Value9876543210==": (
                "AZURE_ACS_EMAIL_CONNECTION_STRING=endpoint=https://valstorm-acs.communication.azure.com/;accesskey="
                + "MockAcsAccessKeyBase64Value9876543210=="
            ),
            "MockServiceBusSharedAccessKey1234567890=": (
                "Endpoint=sb://valstorm.servicebus.windows.net/;SharedAccessKeyName=Root;SharedAccessKey="
                + "MockServiceBusSharedAccessKey1234567890="
            ),
            "MockSasSignatureValue1234567890abcd": (
                "https://valstorm.blob.core.windows.net/docs/a.pdf?sv=2022-11-02&sig="
                + "MockSasSignatureValue1234567890abcd"
            ),
            "val_prod_api_key_9988776655": "x-api-key: " + "val_prod_api_key_9988776655",
            "0123456789abcdef0123456789abcdef": "Ocp-Apim-Subscription-Key: " + "0123456789abcdef0123456789abcdef",
            "SqlAdminPass2026!": "Server=sql.local;User Id=sa;Password=" + "SqlAdminPass2026!" + ";",
            "oauth_secret_val_998877": "https://auth.example.com/token?client_id=abc&client_secret=" + "oauth_secret_val_998877",
        }

        combined = "\n".join(samples.values())
        out = sanitizer.sanitize(combined)

        for secret in samples:
            assert secret not in out, f"Secret '{secret}' leaked:\n{out}"

        assert "postgres://admin:<REDACTED_DB_PASSWORD>@db.internal:5432/valstorm" in out
        assert "rediss://:<REDACTED_DB_PASSWORD>@cache.redis.cache.windows.net:6380/0" in out
        assert "AccountKey=<REDACTED_AZURE_KEY>" in out
        assert "x-api-key: <REDACTED_API_KEY>" in out
        assert "Password=<REDACTED_SECRET>;" in out
        assert "client_secret=<REDACTED_SECRET>" in out
        # Idempotent
        assert sanitizer.sanitize(out) == out

    def test_azure_acs_env_var_masked_from_os_environ(self, monkeypatch):
        acs_val = "endpoint=https://valstorm.communication.azure.com/;accesskey=" + "EnvAcsKey9988776655443322=="
        monkeypatch.setenv("AZURE_ACS_EMAIL_CONNECTION_STRING", acs_val)
        sanitizer = SecretSanitizer(include_env=True)
        out = sanitizer.sanitize(f"Loaded ACS config: {acs_val}")
        assert "EnvAcsKey9988776655443322==" not in out
        assert "<REDACTED_ENV_SECRET:AZURE_ACS_EMAIL_CONNECTION_STRING>" in out

    def test_security_guard_redact_secrets_covers_pci_and_azure(self):
        from core.security_guard import SecurityGuard

        pan = "4111-" + "1111-" + "1111-" + "1111"
        raw = (
            f"Card {pan} CVV: 742 "
            + "x-api-key: " + "secret_header_value_987654 "
            + "DefaultEndpointsProtocol=https;AccountName=a;AccountKey=" + "StorageKeyBase64Value1234567890=="
        )
        out = SecurityGuard.redact_secrets(raw)
        assert pan not in out
        assert "742" not in out
        assert "secret_header_value_987654" not in out
        assert "StorageKeyBase64Value1234567890==" not in out
        assert "<REDACTED_PAYMENT_CARD>" in out
        assert "<REDACTED_CVV>" in out
        assert "<REDACTED_API_KEY>" in out
        assert "<REDACTED_AZURE_KEY>" in out


class TestEgressChokepointsAndPersistence:
    """Verifies redaction across user inputs, parsed CSV/XLSX/PDF tool outputs, providers, and chat logs."""

    @pytest.mark.asyncio
    async def test_react_engine_redacts_user_input_and_parsed_document_tool_outputs(self):
        from core.models import Message, SessionState, ToolCall
        from core.react import ReActEngine

        registry = ToolRegistry()
        pan_csv = "4111-" + "1111-" + "1111-" + "1111"
        pan_pdf = "3782-" + "822463-" + "10005"
        pan_user = "5555-" + "5555-" + "5555-" + "4444"
        pg_pass = "UltraSecretPg99!"

        @registry.register
        def parse_billing_csv() -> str:
            """Simulate parsed CSV / XLSX tabular output."""
            return f"patient_name,card_number,exp_date,cvv,zip\nJane Doe,{pan_csv},09/28,742,90210"

        @registry.register
        def parse_invoice_pdf() -> str:
            """Simulate extracted PDF invoice text."""
            return f"Invoice #4410\nAmex: {pan_pdf}\nExp: 12/2028\nSecurity Code: 8431\nDB: postgres://app:{pg_pass}@db:5432/inv"

        seen_messages = []

        class CapturingProvider:
            provider_name = "mock"

            def __init__(self):
                self.step = 0

            async def generate(self, messages, tools=None, model=None):
                seen_messages.append([m.model_dump() for m in messages])
                self.step += 1
                if self.step == 1:
                    return Message(
                        role="assistant",
                        content="Reading documents...",
                        tool_calls=[
                            ToolCall(id="tc_csv", name="parse_billing_csv", arguments={}),
                            ToolCall(id="tc_pdf", name="parse_invoice_pdf", arguments={}),
                        ],
                    )
                return Message(role="assistant", content="Reviewed redacted billing docs safely.")

        session = SessionState(session_id="test_pci_egress")
        engine = ReActEngine(provider=CapturingProvider(), tools=registry)
        user_prompt = f"Check billing for card {pan_user} CVV: 319 and parse docs."
        await engine.run_turn(session=session, user_input=user_prompt)

        # Every payload transmitted to the LLM provider must be 100% free of raw PCI & DB secrets
        serialized_egress = str(seen_messages)
        for secret in (pan_user, "319", pan_csv, "742", pan_pdf, "8431", pg_pass):
            assert secret not in serialized_egress, f"Raw secret '{secret}' reached LLM provider!"

        assert "<REDACTED_PAYMENT_CARD>" in serialized_egress
        assert "<REDACTED_CVV>" in serialized_egress
        assert "<REDACTED_CARD_EXPIRY>" in serialized_egress
        assert "<REDACTED_DB_PASSWORD>" in serialized_egress

    def test_tool_registry_redacts_error_output(self):
        registry = ToolRegistry()
        pan = "4242-" + "4242-" + "4242-" + "4242"

        @registry.register
        def failing_tool(card: str) -> str:
            """Always raises with sensitive input."""
            raise RuntimeError(f"Gateway rejected card {card} with password=TopSecretError99!")

        res = registry.execute("failing_tool", {"card": pan})
        assert res.is_error is True
        assert pan not in res.output
        assert "TopSecretError99!" not in res.output
        assert "<REDACTED_PAYMENT_CARD>" in res.output
        assert "<REDACTED_SECRET>" in res.output

    def test_session_store_persisted_chat_logs_redacted(self, tmp_path):
        import sqlite3
        from core.models import Message, SessionState, ToolCall, ToolResult
        from core.storage import SessionStore

        db_path = tmp_path / "chat_logs.db"
        store = SessionStore(db_path=db_path)

        pan = "4111-" + "1111-" + "1111-" + "1111"
        pg_pass = "PersistedDbPass99!"
        session = SessionState(session_id="sess_pci_log")
        session.add_message(
            Message(
                role="user",
                content=f"Charge {pan} CVV: 742 using postgres://admin:{pg_pass}@db:5432/prod",
            )
        )
        session.add_message(
            Message(
                role="assistant",
                content="Calling tool",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="pay",
                        arguments={"card": pan, "cvv": "742", "client_secret": "oauth_persisted_998877"},
                    )
                ],
            )
        )
        session.add_message(
            Message(
                role="tool",
                content=f"Result for {pan}",
                tool_result=ToolResult(call_id="tc1", name="pay", output=f"Card {pan} cvc=999"),
            )
        )

        store.save_session(session)

        # Inspect raw SQLite tables directly (ai_chat, ai_chat_message, ai_chat_message_fts)
        conn = sqlite3.connect(str(db_path))
        chat_rows = conn.execute("SELECT name, metadata_json FROM ai_chat").fetchall()
        msg_rows = conn.execute("SELECT body, tool_calls, tool_result FROM ai_chat_message").fetchall()
        fts_rows = conn.execute("SELECT body FROM ai_chat_message_fts").fetchall()
        conn.close()

        raw_dump = f"{chat_rows}\n{msg_rows}\n{fts_rows}"
        for secret in (pan, "742", "999", pg_pass, "oauth_persisted_998877"):
            assert secret not in raw_dump, f"Secret '{secret}' persisted in SQLite chat log!"
        assert "<REDACTED_PAYMENT_CARD>" in raw_dump
        assert "<REDACTED_CVV>" in raw_dump
        assert "<REDACTED_DB_PASSWORD>" in raw_dump

    def test_provider_format_messages_redacts_defense_in_depth(self):
        from core.models import Message
        from providers.openai import OpenAIProvider
        from providers.anthropic import AnthropicProvider
        from providers.gemini import GeminiProvider

        pan = "4111-" + "1111-" + "1111-" + "1111"
        msgs = [
            Message(role="user", content=f"My card is {pan} CVV: 456 x-api-key: " + "provider_key_99887766"),
        ]

        oai_formatted = OpenAIProvider(api_key="test")._format_messages(msgs)
        ant_sys, ant_formatted = AnthropicProvider(api_key="test")._format_messages(msgs)
        gem_sys, gem_contents = GeminiProvider(api_key="test")._format_messages(msgs)

        for dump in (str(oai_formatted), f"{ant_sys}\n{ant_formatted}", f"{gem_sys}\n{gem_contents}"):
            assert pan not in dump
            assert "456" not in dump
            assert "provider_key_99887766" not in dump
            assert "<REDACTED_PAYMENT_CARD>" in dump
            assert "<REDACTED_CVV>" in dump
            assert "<REDACTED_API_KEY>" in dump

    def test_read_local_csv_and_vfs_parsed_xlsx_pdf_redacted_via_tool_registry(self, tmp_path):
        from core.tools import read_local_file

        pan_visa = "4111-" + "1111-" + "1111-" + "1111"
        pan_amex = "3782-" + "822463-" + "10005"
        acs_key = "ParsedXlsxAcsKeyBase64Value123456=="

        csv_file = tmp_path / "billing_export.csv"
        csv_file.write_text(
            f"patient_id,npi,card_number,exp,cvv,zip\npat_4111111111111111,1234567893,{pan_visa},09/28,742,90210\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.register(read_local_file)

        @registry.register
        def mock_vfs_get_file(file_id: str) -> dict:
            """Simulate VFS parsed .xlsx/.pdf JSON payload."""
            return {
                "file_id": file_id,
                "name": "Q4_Billing_Sheet.xlsx",
                "text_content": (
                    f"Row 1: Amex {pan_amex} Exp 11/2029 CVC: 8431\n"
                    f"ACS: endpoint=https://acme.communication.azure.com/;accesskey={acs_key}"
                ),
            }

        csv_res = registry.execute("read_local_file", {"path": str(csv_file)})
        assert csv_res.is_error is False
        assert pan_visa not in csv_res.output
        assert "742" not in csv_res.output
        assert "09/28" not in csv_res.output
        assert "<REDACTED_PAYMENT_CARD>" in csv_res.output
        assert "<REDACTED_CARD_EXPIRY>" in csv_res.output
        assert "<REDACTED_CVV>" in csv_res.output
        # Zero false positives on patient ID, NPI, and ZIP
        assert "pat_4111111111111111" in csv_res.output
        assert "1234567893" in csv_res.output
        assert "90210" in csv_res.output

        vfs_res = registry.execute("mock_vfs_get_file", {"file_id": "file_xlsx_1"})
        assert vfs_res.is_error is False
        assert pan_amex not in vfs_res.output
        assert "8431" not in vfs_res.output
        assert acs_key not in vfs_res.output
        assert "<REDACTED_PAYMENT_CARD>" in vfs_res.output
        assert "<REDACTED_CVV>" in vfs_res.output
        assert "<REDACTED_AZURE_KEY>" in vfs_res.output

    @pytest.mark.asyncio
    async def test_react_engine_redacts_steer_input(self):
        from core.models import Message, SessionState
        from core.react import ReActEngine

        pan = "4242-" + "4242-" + "4242-" + "4242"
        seen = []

        class SteerProvider:
            provider_name = "mock"

            async def generate(self, messages, tools=None, model=None):
                seen.append([m.model_dump() for m in messages])
                return Message(role="assistant", content="Done.")

        session = SessionState(session_id="sess_steer_pci")
        engine = ReActEngine(provider=SteerProvider(), tools=ToolRegistry())
        engine.steer(f"Actually use card {pan} CVV: 918")
        await engine.run_turn(session=session, user_input="Start")

        dump = str(seen)
        assert pan not in dump
        assert "918" not in dump
        assert "<REDACTED_PAYMENT_CARD>" in dump
        assert "<REDACTED_CVV>" in dump

    @pytest.mark.asyncio
    async def test_server_sync_turn_to_valstorm_redacts_payload(self, monkeypatch):
        import server

        captured_payloads = []

        class DummyResp:
            status_code = 200
            text = "ok"

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                captured_payloads.append(json)
                return DummyResp()

        monkeypatch.setattr(server.httpx, "AsyncClient", DummyClient)

        pan = "4111-" + "1111-" + "1111-" + "1111"
        await server._sync_turn_to_valstorm(
            chat_id="chat_123",
            run_id="run_1",
            output_text=f"Processed card {pan} CVV: 742",
            status="completed",
            input_tokens=10,
            output_tokens=20,
            model="gemini-flash-latest",
            provider="gemini",
            valstorm_token="tok_test",
            valstorm_base_url="https://api.example.com",
            user_text=f"Use password=SyncSecret99! and card {pan}",
            error=f"Failed with redis://:RedisSyncPass88!@redis:6379/0",
            tool_calls=[{"name": "charge", "args": {"card": pan, "cvv": "742"}}],
            execution_trace=[{"kind": "narration", "text": f"Card {pan}"}],
        )

        assert len(captured_payloads) == 1
        dump = str(captured_payloads[0])
        for secret in (pan, "742", "SyncSecret99!", "RedisSyncPass88!"):
            assert secret not in dump, f"Secret '{secret}' leaked in desktop-sync payload!"
        assert "<REDACTED_PAYMENT_CARD>" in dump
        assert "<REDACTED_CVV>" in dump
        assert "<REDACTED_SECRET>" in dump
        assert "<REDACTED_DB_PASSWORD>" in dump
