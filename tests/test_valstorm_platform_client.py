"""Tests for RemotePlatformContext and execute_code Platform Binding."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tools.valstorm_client import ValstormApiClient
from tools.valstorm_platform_client import (
    RemotePlatformContext,
    RemoteRecordContext,
    RemoteQueryContext,
    RemoteSchemaContext,
    RemoteFileContext,
)
from tools.execute_code import create_execute_code_tools, _build_execution_environment
from tools.valstorm_tools import create_valstorm_tools


@pytest.fixture
def mock_api_client():
    client = MagicMock(spec=ValstormApiClient)
    client.token = "mock_jwt_token_123"
    client.base_url = "https://api.valstorm.com/v1"
    
    # Mock methods
    client.sql_query = AsyncMock(return_value={"records": [{"id": "cont_1", "name": "Alice"}]})
    client.records_create = AsyncMock(return_value={"id": "cont_2", "name": "Bob"})
    client.records_update = AsyncMock(return_value={"id": "cont_2", "name": "Bob Updated"})
    client.records_delete = AsyncMock(return_value={"status": "success", "deleted_count": 1})
    client.records_merge = AsyncMock(return_value={"master_record": {"id": "cont_1", "name": "Alice"}, "deleted_records": ["cont_2"]})
    client.records_hydrate_batch = AsyncMock(return_value={"cont_1": {"name": "Alice", "schema": "contact"}})
    client.schema_get = AsyncMock(return_value={"title": "Contact", "properties": {"name": {"type": "string"}}})
    client.schema_create = AsyncMock(return_value={"id": "obj_new", "name": "Project"})
    client.vfs_search = AsyncMock(return_value={"results": [{"file_id": "file_1", "filename": "spec.md"}]})
    client.vfs_get_file = AsyncMock(return_value={"id": "file_1", "delivery_type": "inline_text", "text_content": "# Hello VFS"})
    client.vfs_browse_vault = AsyncMock(return_value={"vault_id": "vaul_1", "files": []})
    client.vfs_browse_path = AsyncMock(return_value={"vault_id": "vaul_finance", "files": []})
    client.vfs_get_tree = AsyncMock(return_value={"tree": []})
    client.vfs_get_snapshot = AsyncMock(return_value={"vaults": [], "files": []})
    client.vfs_delete_item = AsyncMock(return_value={"status": "success"})
    return client


def test_remote_platform_context_initialization(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    assert isinstance(platform.records, RemoteRecordContext)
    assert isinstance(platform.query, RemoteQueryContext)
    assert isinstance(platform.schema, RemoteSchemaContext)
    assert isinstance(platform.files, RemoteFileContext)
    assert "RemotePlatformContext" in repr(platform)


def test_remote_platform_query_sql(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    records = platform.query.sql("SELECT id, name FROM contact")
    assert records == [{"id": "cont_1", "name": "Alice"}]
    mock_api_client.sql_query.assert_called_once_with(query="SELECT id, name FROM contact LIMIT 50", bypass_cache=False)
    assert not hasattr(platform.query, "mongo")


def test_remote_platform_records_crud(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    
    # Create
    c_res = platform.records.create("contact", {"name": "Bob"})
    assert c_res["name"] == "Bob"
    mock_api_client.records_create.assert_called_once_with(api_name="contact", records=[{"name": "Bob"}])

    # Update
    u_res = platform.records.update("contact", {"id": "cont_2", "name": "Bob Updated"})
    assert u_res["name"] == "Bob Updated"
    mock_api_client.records_update.assert_called_once_with(api_name="contact", records=[{"id": "cont_2", "name": "Bob Updated"}])

    # Delete
    d_res = platform.records.delete("contact", "cont_2")
    assert d_res["deleted_count"] == 1
    mock_api_client.records_delete.assert_called_once_with(api_name="contact", ids=["cont_2"])

    # Merge
    m_res = platform.records.merge(master_id="cont_1", duplicate_ids=["cont_2"], schema_api_name="contact")
    assert m_res["master_record"]["id"] == "cont_1"
    mock_api_client.records_merge.assert_called_once_with(
        master_id="cont_1",
        duplicate_ids=["cont_2"],
        schema_api_name="contact",
        field_overrides=None,
    )

    # Hydrate batch
    h_res = platform.records.hydrate_batch(["cont_1"])
    assert "cont_1" in h_res


def test_remote_platform_vfs_file_and_schema(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    
    # VFS Get File
    f_res = platform.files.get_file("file_1")
    assert f_res["text_content"] == "# Hello VFS"
    mock_api_client.vfs_get_file.assert_called_once_with(file_id="file_1")

    # Schema Create
    s_res = platform.schema.create({"name": "Project", "app": "crm"})
    assert s_res["id"] == "obj_new"


@pytest.mark.asyncio
async def test_valstorm_direct_tools(mock_api_client):
    tools = {t.__name__: t for t in create_valstorm_tools(client=mock_api_client)}
    assert "valstorm_mongo_query" not in tools
    assert "valstorm_vfs_get_file" in tools
    assert "valstorm_records_hydrate_batch" in tools

    res_file = await tools["valstorm_vfs_get_file"](file_id="file_1")
    assert "Hello VFS" in res_file


@pytest.mark.asyncio
async def test_execute_code_with_bound_platform(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    execute_tools = create_execute_code_tools(client=mock_api_client, platform=platform)
    execute_code_fn = execute_tools[0]

    code = """
results = platform.query.sql("SELECT id, name FROM contact")
doc = platform.files.get_file("file_1")
print(f"Found: {results[0]['name']} and doc: {doc['text_content']}")
"""
    output = await execute_code_fn(code=code)
    assert "Found: Alice and doc: # Hello VFS" in output


@pytest.mark.asyncio
async def test_execute_code_multistep_script(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    execute_tools = create_execute_code_tools(client=mock_api_client, platform=platform)
    execute_code_fn = execute_tools[0]

    code = """
created = platform.records.create("contact", {"name": "Charlie"})
updated = platform.records.update("contact", {"id": created["id"], "status": "Active"})
print("Workflow completed successfully")
"""
    output = await execute_code_fn(code=code)
    assert "Workflow completed successfully" in output


def test_valstorm_api_client_recreates_http_client_across_closed_event_loops():
    """Calling ValstormApiClient across separate asyncio.run() loops (as execute_code does for
    consecutive records_create / records_update calls) must recreate the owned httpx.AsyncClient
    when the event loop changes instead of failing with RuntimeError: Event loop is closed."""
    import httpx

    api_client = ValstormApiClient(token="tok_test_loop", base_url="https://api.valstorm.com/v1")

    seen_clients: list[httpx.AsyncClient] = []

    async def _probe_client() -> httpx.AsyncClient:
        c = api_client._get_active_client()
        seen_clients.append(c)
        return c

    c1 = asyncio.run(_probe_client())
    # First asyncio.run loop is now closed; a second asyncio.run loop must get a fresh client
    c2 = asyncio.run(_probe_client())

    assert c1 is not c2, "Expected fresh httpx.AsyncClient when running on a new event loop"
    assert not c2.is_closed

    # Same loop must reuse the same client instance
    async def _probe_twice_same_loop():
        a = api_client._get_active_client()
        b = api_client._get_active_client()
        await api_client.close()
        return a, b

    a, b = asyncio.run(_probe_twice_same_loop())
    assert a is b


def test_valstorm_api_client_loop_safe_refresh_lock():
    """ValstormApiClient must provide an initialized loop-aware asyncio.Lock for proactive refresh."""
    api_client = ValstormApiClient(token="tok_test_lock", base_url="https://api.valstorm.com/v1")

    async def _get_lock():
        lock = api_client._get_lock()
        async with lock:
            return lock

    lock1 = asyncio.run(_get_lock())
    lock2 = asyncio.run(_get_lock())
    assert isinstance(lock1, asyncio.Lock)
    assert isinstance(lock2, asyncio.Lock)
    assert lock1 is not lock2


@pytest.mark.asyncio
async def test_execute_code_functions_can_reference_top_level_vars_and_imports(mock_api_client):
    """Top-level imports and variables defined inside an execute_code script must be visible
    inside functions and comprehensions defined in the same script (unified globals/locals namespace)."""
    platform = RemotePlatformContext(client=mock_api_client)
    execute_tools = create_execute_code_tools(client=mock_api_client, platform=platform)
    execute_code_fn = execute_tools[0]

    code = """
import math as _m

payload = {"label": "valstorm", "val": 16}

def compute():
    return f"{payload['label']}:{int(_m.sqrt(payload['val']))}"

print(compute())
"""
    output = await execute_code_fn(code=code)
    assert "valstorm:4" in output, f"Unexpected output: {output}"


@pytest.mark.asyncio
async def test_valstorm_client_analytics_v2_and_dual_format_generators():
    client = ValstormApiClient(token="tok_123", base_url="http://127.0.0.1:8010/v1")

    # 1. analytics_query_v2 strips /v1 and hits /v2/analytics/query
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"columns": [], "rows": [], "totals": {"cnt": 42}}
    client._request_with_retry = AsyncMock(return_value=fake_resp)

    res = await client.analytics_query_v2(
        {"source": "deal", "dimensions": [], "measures": [{"agg": "count", "alias": "cnt"}]}
    )
    assert res["totals"]["cnt"] == 42
    client._request_with_retry.assert_awaited_once_with(
        "POST",
        "http://127.0.0.1:8010/v2/analytics/query",
        json={"source": "deal", "dimensions": [], "measures": [{"agg": "count", "alias": "cnt"}]},
    )

    # 2. generate_report emits both V2 native specs and V1 legacy keys
    client.schema_get = AsyncMock(return_value={"id": "obj_deal", "api_name": "deal"})
    client.records_create = AsyncMock(return_value=[{"id": "rep_v2_1", "name": "Monthly Revenue"}])
    await client.generate_report(
        name="Monthly Revenue",
        object_api_name="deal",
        sql_query="SELECT * FROM deal WHERE stage = 'Won'",
        chart_type="Area",
        x_field="close_date",
        y_field="amount",
        aggregation="Sum",
        group_by_time="Month",
    )
    _, rep_records = client.records_create.call_args.args
    rep_payload = rep_records[0]
    assert rep_payload["query"]["version"] == 2
    assert rep_payload["query"]["v2_query"]["source"] == "deal"
    assert rep_payload["query"]["v2_query"]["dimensions"] == [
        {"field": "close_date", "time_grain": "month", "fill_gaps": True}
    ]
    assert rep_payload["query"]["v2_query"]["measures"][0]["agg"] == "sum"
    assert rep_payload["chart_settings"]["v2_spec"]["type"] == "area"
    # Legacy keys still preserved
    assert rep_payload["chart_settings"]["chartType"] == "Area"

    # 3. generate_dashboard emits __v2_config__ with live linked_report bindings
    client.records_create = AsyncMock(
        side_effect=[
            [{"id": "rep_auto_1", "name": "Pipeline KPI"}],
            [{"id": "dash_v2_1", "name": "Exec Board"}],
        ]
    )
    dash_out = await client.generate_dashboard(
        name="Exec Board",
        widgets=[
            {
                "title": "Pipeline KPI",
                "object_api_name": "deal",
                "chart_type": "Metric",
                "y_field": "amount",
                "aggregation": "Sum",
            }
        ],
    )
    assert dash_out["id"] == "dash_v2_1"
    dash_call_records = client.records_create.call_args_list[1].args[1]
    layout_items = dash_call_records[0]["layout_items"]
    assert layout_items[0]["i"] == "__v2_config__"
    v2_cfg = layout_items[0]["v2_config"]
    assert v2_cfg["version"] == 2
    assert v2_cfg["widgets"][0]["kind"] == "linked_report"
    assert v2_cfg["widgets"][0]["reportId"] == "rep_auto_1"
