"""High-performance Web Scraping, Search, Domain Crawling, and Content Diff tools for vsagent."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx

from core.tools import ToolRegistry, tool

logger = logging.getLogger("vsagent.web_tools")

DEFAULT_SCRAPER_SERVICE_URL = os.environ.get("WEB_SCRAPER_SERVICE_URL", "http://127.0.0.1:8661/v1")


class WebScraperClient:
    """Client for the Valstorm Web Intelligence & Scraping Microservice."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or DEFAULT_SCRAPER_SERVICE_URL).rstrip("/")

    async def scrape(
        self,
        url: str,
        force_refresh: bool = False,
        include_html: bool = False,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrapes or retrieves a web page using 3-tier waterfall and zstd cache."""
        endpoint = f"{self.base_url}/scrape"
        payload = {
            "url": url,
            "force_refresh": force_refresh,
            "include_html": include_html,
            "tenant_id": tenant_id
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(endpoint, json=payload)
            res.raise_for_status()
            return res.json()

    async def search(
        self,
        query: str,
        limit: int = 5,
        auto_scrape_top: int = 0,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Searches the live web and optionally auto-scrapes top N hits."""
        endpoint = f"{self.base_url}/search"
        payload = {
            "query": query,
            "limit": limit,
            "auto_scrape_top": auto_scrape_top,
            "tenant_id": tenant_id
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(endpoint, json=payload)
            res.raise_for_status()
            return res.json()

    async def crawl(
        self,
        url: str,
        max_pages: int = 10,
        max_concurrency: int = 5,
        force_refresh: bool = False,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crawls up to max_pages across a target customer domain."""
        endpoint = f"{self.base_url}/crawl"
        payload = {
            "url": url,
            "max_pages": max_pages,
            "max_concurrency": max_concurrency,
            "force_refresh": force_refresh,
            "tenant_id": tenant_id
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(endpoint, json=payload)
            res.raise_for_status()
            return res.json()

    async def diff(
        self,
        url: str,
        force_live_check: bool = True,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compares live webpage content with previous archive to detect changes."""
        endpoint = f"{self.base_url}/diff"
        payload = {
            "url": url,
            "force_live_check": force_live_check,
            "tenant_id": tenant_id
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(endpoint, json=payload)
            res.raise_for_status()
            return res.json()

    async def poll_feed(
        self,
        feed_url: str,
        auto_scrape_articles: bool = True,
        max_articles: int = 10,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Polls an RSS or Atom feed and extracts clean Markdown articles."""
        endpoint = f"{self.base_url}/feeds/poll"
        payload = {
            "feed_url": feed_url,
            "auto_scrape_articles": auto_scrape_articles,
            "max_articles": max_articles,
            "tenant_id": tenant_id
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(endpoint, json=payload)
            res.raise_for_status()
            return res.json()


def create_web_tools(client: Optional[WebScraperClient] = None) -> List[Any]:
    """Creates Web Intelligence tool functions bound to a WebScraperClient."""
    web_client = client or WebScraperClient()

    @tool
    async def web_scrape(
        url: str,
        force_refresh: bool = False,
        include_html: bool = False
    ) -> str:
        """Fetches and converts any public webpage into clean, token-optimized Markdown.

        Uses a 3-tier waterfall pipeline (Fast Async HTTP -> Stealth Headless Playwright -> Firecrawl API)
        and caches results permanently with Python 3.14 native zstd compression.

        Args:
            url: The full web URL to scrape (e.g. 'https://stripe.com/pricing' or 'https://acme.com').
            force_refresh: If True, bypasses cache and performs a fresh live crawl.
            include_html: If True, includes raw HTML in addition to Markdown.
        """
        try:
            res = await web_client.scrape(url=url, force_refresh=force_refresh, include_html=include_html)
            title = res.get("title") or "Untitled"
            cached_flag = " (Cached hit)" if res.get("cached") else f" (Tier: {res.get('tier_used')})"
            canonical = res.get("canonical_url") or url
            md_content = res.get("markdown") or "No textual content found."
            
            # Security: Wrap in untrusted data boundary tags to prevent prompt injection
            return (
                f"<untrusted_web_content url=\"{canonical}\" title=\"{title}\" source=\"web_scrape\"{cached_flag}>\n"
                f"{md_content}\n"
                f"</untrusted_web_content>"
            )
        except Exception as e:
            return f"Error scraping {url}: {e}"

    @tool
    async def web_search(
        query: str,
        limit: int = 5,
        auto_scrape_top: int = 0
    ) -> str:
        """Executes a live search query across the public web and returns snippets and URLs.

        Args:
            query: Search query or question (e.g. 'top MSP backup solutions 2026' or 'who is CEO of Acme Corp').
            limit: Maximum number of search result hits to return (default: 5).
            auto_scrape_top: If greater than 0, automatically fetches and includes full AI-cleaned Markdown for top N hits.
        """
        try:
            res = await web_client.search(query=query, limit=limit, auto_scrape_top=auto_scrape_top)
            hits = res.get("hits", [])
            if not hits:
                return f"No search results found for query: '{query}'"

            output = [f"### Web Search Results for: '{query}'\n"]
            for i, h in enumerate(hits, 1):
                output.append(f"{i}. **[{h.get('title')}]({h.get('url')})**\n   {h.get('snippet')}\n")

            scraped = res.get("scraped_contents", [])
            if scraped:
                output.append("\n---\n### Scraped Content Excerpts:\n")
                for s in scraped:
                    output.append(f"#### {s.get('title')} ({s.get('url')})\n{s.get('markdown')}\n")

            return "\n".join(output)
        except Exception as e:
            return f"Error executing web search for '{query}': {e}"

    @tool
    async def web_crawl_domain(
        url: str,
        max_pages: int = 10,
        force_refresh: bool = False
    ) -> str:
        """Crawls multiple internal pages across a company or customer website.

        Discovers internal links, ignores media/noise, and aggregates clean Markdown for each page.

        Args:
            url: Root website URL to begin crawling (e.g. 'https://customer.com').
            max_pages: Maximum number of internal pages to crawl (default: 10, max: 25).
            force_refresh: If True, bypasses previously cached pages.
        """
        try:
            res = await web_client.crawl(url=url, max_pages=max_pages, force_refresh=force_refresh)
            pages = res.get("pages", [])
            if not pages:
                return f"No crawlable pages found for domain: {url}"

            summary = [f"### Domain Crawl Summary: {res.get('domain')} ({res.get('total_pages_crawled')} pages crawled)\n"]
            for i, p in enumerate(pages, 1):
                title = p.get("title") or "Page"
                summary.append(f"### Page {i}: {title} ({p.get('url')})\n{p.get('markdown')}\n---")

            return "\n\n".join(summary)
        except Exception as e:
            return f"Error crawling domain {url}: {e}"

    @tool
    async def web_content_diff(
        url: str,
        force_live_check: bool = True
    ) -> str:
        """Compares the live version of a webpage against its permanently archived version to detect changes.

        Args:
            url: Target web URL (e.g. 'https://competitor.com/pricing').
            force_live_check: If True, performs a fresh live scrape to compare against archive.
        """
        try:
            res = await web_client.diff(url=url, force_live_check=force_live_check)
            if not res.get("has_changed"):
                return f"✅ No changes detected for {url} (Similarity: 100%). Content is identical to archived version from {res.get('previous_scraped_at')}."

            similarity = res.get("similarity_score", 0.0) * 100
            diff_text = res.get("unified_diff") or "Content modified."
            return (
                f"⚠️ Content changes detected for {url}!\n"
                f"Similarity: {similarity:.1f}%\n"
                f"Added lines: {res.get('added_lines_count', 0)}, Removed lines: {res.get('removed_lines_count', 0)}\n\n"
                f"### Unified Diff:\n```diff\n{diff_text}\n```"
            )
        except Exception as e:
            return f"Error computing content diff for {url}: {e}"

    @tool
    async def web_feed_poll(
        feed_url: str,
        max_articles: int = 10
    ) -> str:
        """Polls an RSS 2.0 or Atom news feed and retrieves recent articles with clean Markdown text.

        Args:
            feed_url: XML RSS/Atom feed URL (e.g. 'https://news.ycombinator.com/rss' or 'https://techcrunch.com/feed/').
            max_articles: Maximum number of recent articles to return (default: 10).
        """
        try:
            res = await web_client.poll_feed(feed_url=feed_url, auto_scrape_articles=True, max_articles=max_articles)
            items = res.get("items", [])
            if not items:
                return f"No articles found in feed: {feed_url}"

            output = [f"### News Feed: {feed_url} ({len(items)} items)\n"]
            for i, it in enumerate(items, 1):
                output.append(f"{i}. **{it.get('title')}**\n   Link: {it.get('link')}\n   Published: {it.get('pub_date') or 'N/A'}\n   Summary: {it.get('description')}\n")

            return "\n".join(output)
        except Exception as e:
            return f"Error polling feed {feed_url}: {e}"

    return [
        web_scrape,
        web_search,
        web_crawl_domain,
        web_content_diff,
        web_feed_poll,
    ]


# Export default tool instances
_default_tools = create_web_tools()
web_scrape = _default_tools[0]
web_search = _default_tools[1]
web_crawl_domain = _default_tools[2]
web_content_diff = _default_tools[3]
web_feed_poll = _default_tools[4]


def register_web_tools(registry: ToolRegistry, client: Optional[WebScraperClient] = None) -> ToolRegistry:
    """Registers all Web Intelligence tools into the provided ToolRegistry."""
    tools = create_web_tools(client)
    for tool_fn in tools:
        registry.register(tool_fn)
    return registry
