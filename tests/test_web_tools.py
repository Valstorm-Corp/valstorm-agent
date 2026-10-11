import pytest
from core.tools import ToolRegistry
from tools.web_tools import (
    WebScraperClient,
    create_web_tools,
    register_web_tools,
)


@pytest.mark.asyncio
async def test_web_tools_registration():
    registry = ToolRegistry()
    register_web_tools(registry)

    assert registry.get("web_scrape") is not None
    assert registry.get("web_search") is not None
    assert registry.get("web_crawl_domain") is not None
    assert registry.get("web_content_diff") is not None
    assert registry.get("web_feed_poll") is not None


@pytest.mark.asyncio
async def test_web_scrape_tool_mock(monkeypatch):
    mock_result = {
        "url": "https://example.com/pricing",
        "canonical_url": "https://example.com/pricing",
        "domain": "example.com",
        "title": "Example Pricing",
        "markdown": "# Pricing\n\n- Basic: $10\n- Enterprise: $50",
        "tier_used": "fast_http",
        "cached": True
    }

    async def mock_scrape(self, url, **kwargs):
        return mock_result

    monkeypatch.setattr(WebScraperClient, "scrape", mock_scrape)

    tools = create_web_tools(WebScraperClient())
    web_scrape_fn = tools[0]

    output = await web_scrape_fn(url="https://example.com/pricing")
    assert "Example Pricing" in output
    assert "Cached hit" in output
    assert "Enterprise: $50" in output


@pytest.mark.asyncio
async def test_web_search_tool_mock(monkeypatch):
    mock_search_result = {
        "query": "best msp tools",
        "total_hits": 2,
        "hits": [
            {"title": "MSP Tool 1", "url": "https://msp1.com", "snippet": "Leading backup software."},
            {"title": "MSP Tool 2", "url": "https://msp2.com", "snippet": "RMM and Helpdesk platform."}
        ],
        "scraped_contents": []
    }

    async def mock_search(self, query, **kwargs):
        return mock_search_result

    monkeypatch.setattr(WebScraperClient, "search", mock_search)

    tools = create_web_tools(WebScraperClient())
    web_search_fn = tools[1]

    output = await web_search_fn(query="best msp tools")
    assert "MSP Tool 1" in output
    assert "https://msp1.com" in output
    assert "Leading backup software." in output


@pytest.mark.asyncio
async def test_web_content_diff_tool_mock(monkeypatch):
    mock_diff_result = {
        "url": "https://example.com/pricing",
        "has_changed": True,
        "similarity_score": 0.85,
        "added_lines_count": 2,
        "removed_lines_count": 1,
        "unified_diff": "- Basic: $10\n+ Basic: $15"
    }

    async def mock_diff(self, url, **kwargs):
        return mock_diff_result

    monkeypatch.setattr(WebScraperClient, "diff", mock_diff)

    tools = create_web_tools(WebScraperClient())
    web_diff_fn = tools[3]

    output = await web_diff_fn(url="https://example.com/pricing")
    assert "Content changes detected" in output
    assert "Similarity: 85.0%" in output
    assert "+ Basic: $15" in output
