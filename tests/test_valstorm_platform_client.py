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
    client.mongo_query = AsyncMock(return_value={"records": [{"id": "deal_1", "stage": "Closed Won"}]})
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


def test_remote_platform_query_mongo(mock_api_client):
    platform = RemotePlatformContext(client=mock_api_client)
    records = platform.query.mongo("deal", [{"$match": {"stage": "Closed Won"}}])
    assert records == [{"id": "deal_1", "stage": "Closed Won"}]
    mock_api_client.mongo_query.assert_called_once_with(collection="deal", pipeline=[{"$match": {"stage": "Closed Won"}}])


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
    assert "valstorm_mongo_query" in tools
    assert "valstorm_vfs_get_file" in tools
    assert "valstorm_records_hydrate_batch" in tools

    res_file = await tools["valstorm_vfs_get_file"](file_id="file_1")
    assert "Hello VFS" in res_file

    res_mongo = await tools["valstorm_mongo_query"](collection="deal", pipeline=[{"$match": {}}])
    assert "Closed Won" in res_mongo


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
