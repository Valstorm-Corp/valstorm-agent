"""Unit tests for ValstormApiClient with mocked HTTP responses."""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from tools.valstorm_client import ValstormApiClient, resolve_valstorm_credentials


@pytest.mark.asyncio
async def test_valstorm_client_sql_query():
    mock_response = httpx.Response(
        status_code=200,
        json={"records": [{"id": "cont_1", "name": "John Doe", "email": "john@example.com"}]},
        headers={"X-Total-Count": "1"},
        request=httpx.Request("POST", "http://localhost:8000/query"),
    )

    mock_transport = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

    client = httpx.AsyncClient(transport=mock_transport, base_url="http://localhost:8000")
    api_client = ValstormApiClient(token="test_jwt_token", client=client)

    res = await api_client.sql_query("SELECT id, name FROM contact LIMIT 1")
    assert len(res["records"]) == 1
    assert res["records"][0]["name"] == "John Doe"
    assert res["headers"]["x-total-count"] == "1"


@pytest.mark.asyncio
async def test_valstorm_client_vfs_search():
    mock_response = httpx.Response(
        status_code=200,
        json={
            "query": "financials",
            "results": [{"file_id": "file_101", "filename": "Q3_Report.pdf", "score": 0.95}],
        },
        request=httpx.Request("POST", "http://localhost:8000/v1/search"),
    )

    mock_transport = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

    client = httpx.AsyncClient(transport=mock_transport, base_url="http://localhost:8000")
    api_client = ValstormApiClient(token="test_jwt_token", client=client)

    res = await api_client.vfs_search(query="financials", limit=5)
    assert len(res["results"]) == 1
    assert res["results"][0]["filename"] == "Q3_Report.pdf"


@pytest.mark.asyncio
async def test_valstorm_client_records_cud():
    mock_create_resp = httpx.Response(
        status_code=200,
        json=[{"id": "cont_100", "name": "New Contact"}],
        request=httpx.Request("POST", "http://localhost:8000/object/contact"),
    )
    mock_update_resp = httpx.Response(
        status_code=200,
        json=[{"id": "cont_100", "status": "Active"}],
        request=httpx.Request("PATCH", "http://localhost:8000/object/contact"),
    )
    mock_delete_resp = httpx.Response(
        status_code=200,
        json={"message": "deleted"},
        request=httpx.Request("DELETE", "http://localhost:8000/object/contact"),
    )

    mock_transport = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(
        side_effect=[mock_create_resp, mock_update_resp, mock_delete_resp]
    )

    client = httpx.AsyncClient(transport=mock_transport, base_url="http://localhost:8000")
    api_client = ValstormApiClient(token="test_jwt_token", client=client)

    # Create
    created = await api_client.records_create("contact", [{"name": "New Contact"}])
    assert created[0]["id"] == "cont_100"

    # Update
    updated = await api_client.records_update("contact", [{"id": "cont_100", "status": "Active"}])
    assert updated[0]["status"] == "Active"

    # Delete
    deleted = await api_client.records_delete("contact", ["cont_100"])
    assert deleted["deleted_count"] == 1


@pytest.mark.asyncio
async def test_valstorm_client_schema_get():
    mock_response = httpx.Response(
        status_code=200,
        json={"contact": {"title": "Contact", "properties": {"name": {"type": "string"}}}},
        request=httpx.Request("GET", "http://localhost:8000/schema"),
    )

    mock_transport = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

    client = httpx.AsyncClient(transport=mock_transport, base_url="http://localhost:8000")
    api_client = ValstormApiClient(token="test_jwt_token", client=client)

    schemas = await api_client.schema_get()
    assert "contact" in schemas
    assert schemas["contact"]["title"] == "Contact"


def test_jwt_expiry_and_decoding():
    import base64
    import json
    import time
    from tools.valstorm_client import decode_jwt_payload, is_jwt_expired

    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
    exp_time = time.time() - 100
    expired_payload = base64.urlsafe_b64encode(json.dumps({"exp": exp_time, "user": "u1"}).encode()).decode().rstrip("=")
    valid_payload = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + 3600, "user": "u1"}).encode()).decode().rstrip("=")

    expired_jwt = f"{header}.{expired_payload}.signature"
    valid_jwt = f"{header}.{valid_payload}.signature"

    assert decode_jwt_payload(expired_jwt)["user"] == "u1"
    assert is_jwt_expired(expired_jwt) is True
    assert is_jwt_expired(valid_jwt) is False
    assert is_jwt_expired("static-pat-key-12345") is False


def test_refresh_valstorm_tokens_sync_and_async(tmp_path, monkeypatch):
    import json
    import httpx
    from unittest.mock import MagicMock, AsyncMock
    from tools.valstorm_client import refresh_valstorm_tokens_sync, refresh_valstorm_tokens_async

    auth_file = tmp_path / "auth_test.json"
    auth_file.write_text(json.dumps({"access_token": "old_token", "refresh_token": "ref_123"}))

    mock_resp = httpx.Response(200, json={"access_token": "new_access_token", "token_type": "bearer"}, request=httpx.Request("POST", "http://localhost/oauth2/refresh"))

    # Test sync
    orig_post = httpx.Client.post
    monkeypatch.setattr(httpx.Client, "post", MagicMock(return_value=mock_resp))
    res_sync = refresh_valstorm_tokens_sync(
        base_url="https://api.valstorm.com/v1",
        refresh_token="ref_123",
        auth_file_path=auth_file,
    )
    assert res_sync is not None
    new_acc, new_ref = res_sync
    assert new_acc == "new_access_token"
    assert new_ref == "ref_123"

    saved = json.loads(auth_file.read_text())
    assert saved["access_token"] == "new_access_token"


@pytest.mark.asyncio
async def test_refresh_valstorm_tokens_async(tmp_path, monkeypatch):
    import json
    import httpx
    from unittest.mock import AsyncMock
    from tools.valstorm_client import refresh_valstorm_tokens_async

    auth_file = tmp_path / "auth_test_async.json"
    auth_file.write_text(json.dumps({"access_token": "old_token", "refresh_token": "ref_456"}))

    mock_resp = httpx.Response(200, json={"access_token": "new_async_token"}, request=httpx.Request("POST", "http://localhost/oauth2/refresh"))
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=mock_resp))

    res_async = await refresh_valstorm_tokens_async(
        base_url="https://api.valstorm.com/v1",
        refresh_token="ref_456",
        auth_file_path=auth_file,
    )
    assert res_async is not None
    assert res_async[0] == "new_async_token"
    saved = json.loads(auth_file.read_text())
    assert saved["access_token"] == "new_async_token"


def test_resolve_valstorm_auth_context_proactive_refresh(tmp_path, monkeypatch):
    import json
    import time
    import base64
    from tools.valstorm_client import resolve_valstorm_auth_context

    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
    exp_time = time.time() - 100
    expired_payload = base64.urlsafe_b64encode(json.dumps({"exp": exp_time, "user": "u1"}).encode()).decode().rstrip("=")
    expired_jwt = f"{header}.{expired_payload}.signature"

    valstorm_dir = tmp_path / ".valstorm"
    valstorm_dir.mkdir()
    auth_file = valstorm_dir / "auth_prod_default.json"
    auth_file.write_text(json.dumps({"access_token": expired_jwt, "refresh_token": "valid_refresh_token"}))

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "tools.valstorm_client.refresh_valstorm_tokens_sync",
        lambda base_url, refresh_token, auth_file_path: ("refreshed_fresh_access_token", refresh_token),
    )

    token, base_url, refresh_tok, used_file = resolve_valstorm_auth_context(env="prod", profile="default")
    assert token == "refreshed_fresh_access_token"
    assert refresh_tok == "valid_refresh_token"

