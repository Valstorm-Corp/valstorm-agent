"""Unit tests for GitHub Plugin Tools registered in Agent Runtime ToolRegistry."""

import json
from unittest.mock import AsyncMock

import pytest

from core.tools import ToolRegistry
from tools.github_tools import create_github_tools, register_github_tools
from tools.valstorm_tools import register_valstorm_tools


@pytest.fixture
def mock_github_api_client():
    client = AsyncMock()

    async def fake_invoke(target: str, *args, **kwargs):
        if target == "github/list_commits":
            return [
                {
                    "sha": "c333",
                    "commit_title": "Merge pull request #120 from valstorm/FI-584",
                    "is_merge": True,
                },
                {
                    "sha": "c222",
                    "commit_title": "feat(azure): multi-subscription discovery",
                    "is_merge": False,
                },
                {
                    "sha": "c111",
                    "commit_title": "fix(github): add OAuth user scope",
                    "is_merge": False,
                },
            ]
        if target == "github/create_pull_request":
            return {"number": 125, "title": kwargs.get("title"), "state": "open"}
        if target == "github/get_file_contents":
            return {"path": kwargs.get("path"), "decoded_content": "# Test Content"}
        return {"target": target, "kwargs": kwargs}

    client.invoke_platform_method = AsyncMock(side_effect=fake_invoke)
    return client


@pytest.mark.asyncio
async def test_github_tools_registration_and_execution(mock_github_api_client):
    registry = ToolRegistry()
    register_github_tools(registry=registry, client=mock_github_api_client)

    tool_names = registry.list_tools()
    assert "github_list_repositories" in tool_names
    assert "github_list_commits" in tool_names
    assert "github_list_repository_activity" in tool_names
    assert "github_list_pull_requests" in tool_names
    assert "github_create_pull_request" in tool_names
    assert "github_merge_pull_request" in tool_names
    assert "github_get_file_contents" in tool_names
    assert "github_create_or_update_file" in tool_names
    assert "github_list_workflow_runs" in tool_names

    commits_res = await registry.execute_async(
        "github_list_commits",
        {"owner": "valstorm", "repo": "monorepo", "per_page": 3},
    )
    assert not commits_res.is_error
    parsed = json.loads(commits_res.output)
    assert len(parsed) == 3
    assert parsed[0]["commit_title"] == "Merge pull request #120 from valstorm/FI-584"
    assert parsed[0]["is_merge"] is True

    pr_res = await registry.execute_async(
        "github_create_pull_request",
        {"owner": "valstorm", "repo": "monorepo", "title": "FI-585 Expand GitHub Plugin", "head": "FI-585"},
    )
    assert not pr_res.is_error
    assert json.loads(pr_res.output)["number"] == 125


def test_github_tools_included_in_valstorm_tools(mock_github_api_client):
    registry = ToolRegistry()
    register_valstorm_tools(registry=registry, client=mock_github_api_client)
    names = registry.list_tools()
    assert "github_list_commits" in names
    assert "github_list_pull_requests" in names
    assert len( create_github_tools(client=mock_github_api_client) ) >= 30
