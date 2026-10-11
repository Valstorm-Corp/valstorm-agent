"""Client-Side RemotePlatformContext for Valstorm Agent Runtime & External SDK.

Provides 1:1 parity with server-side PlatformContext signatures, delegating operations
over HTTP/REST and WebSocket to the Valstorm API using the user's authenticated session.
"""

import asyncio
from datetime import datetime, timezone
import inspect
import json
from typing import Any, Dict, List, Optional, Union

from tools.valstorm_client import ValstormApiClient


def _make_sync_async_callable(fn):
    """Wraps an async method so it can be called both synchronously and asynchronously."""
    async def _async_wrapper(*args, **kwargs):
        return await fn(*args, **kwargs)

    def _sync_wrapper(*args, **kwargs):
        coro = fn(*args, **kwargs)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # Attach docstring and name
    _sync_wrapper.__name__ = getattr(fn, "__name__", "remote_method")
    _sync_wrapper.__doc__ = getattr(fn, "__doc__", "")
    _sync_wrapper.__is_sync_async_bridge__ = True
    return _sync_wrapper


class RemoteBaseContext:
    """Base class for all remote domain-specific contexts."""
    def __init__(self, platform: 'RemotePlatformContext'):
        self.platform = platform
        self.client: ValstormApiClient = platform.client


class RemoteRecordContext(RemoteBaseContext):
    """Context for record-related operations (CUD)."""

    def create(self, api_name: str, input_data: Union[dict, list[dict]], **kwargs) -> Any:
        """Create one or more records in a collection."""
        records = [input_data] if isinstance(input_data, dict) else input_data
        coro = self.client.records_create(api_name=api_name, records=records)
        return _make_sync_async_callable(lambda: coro)()

    def update(self, api_name: str, input_data: Union[dict, list[dict]], **kwargs) -> Any:
        """Update one or more records in a collection (must contain 'id')."""
        records = [input_data] if isinstance(input_data, dict) else input_data
        coro = self.client.records_update(api_name=api_name, records=records)
        return _make_sync_async_callable(lambda: coro)()

    def delete(self, api_name: str, input_data: Union[dict, list[dict], str, list[str]], **kwargs) -> Any:
        """Delete one or more records by ID or list of record dicts."""
        if isinstance(input_data, str):
            ids = [input_data]
        elif isinstance(input_data, dict):
            ids = [str(input_data.get("id"))]
        elif isinstance(input_data, list):
            ids = [str(x.get("id") if isinstance(x, dict) else x) for x in input_data if x]
        else:
            ids = []
        coro = self.client.records_delete(api_name=api_name, ids=ids)
        return _make_sync_async_callable(lambda: coro)()

    def hydrate_batch(self, ids: List[str]) -> Dict[str, Any]:
        """Batch resolves prefixed IDs (e.g. ['cont_123', 'task_456']) into names and schemas."""
        coro = self.client.records_hydrate_batch(ids=ids)
        return _make_sync_async_callable(lambda: coro)()

    def merge(
        self,
        master_id: str,
        duplicate_ids: Union[str, List[str]],
        schema_api_name: Optional[str] = None,
        field_overrides: Optional[dict] = None,
    ) -> Any:
        """Merges one or more duplicate records into master record and re-links related lookup references."""
        coro = self.client.records_merge(
            master_id=master_id,
            duplicate_ids=duplicate_ids,
            schema_api_name=schema_api_name,
            field_overrides=field_overrides,
        )
        return _make_sync_async_callable(lambda: coro)()


class RemoteQueryContext(RemoteBaseContext):
    """Context for data querying (SQL, Mongo, GraphQL)."""

    def sql(self, query: str, limit: int = 50, bypass_cache: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """Execute a SQL query against the Valstorm query engine."""
        clean_q = query.strip()
        if "limit" not in clean_q.lower() and limit:
            clean_q += f" LIMIT {limit}"

        async def _run():
            res = await self.client.sql_query(query=clean_q, bypass_cache=bypass_cache)
            return res.get("records", [])

        return _make_sync_async_callable(_run)()

    def mongo(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a direct MongoDB aggregation pipeline."""
        async def _run():
            res = await self.client.mongo_query(collection=collection, pipeline=pipeline)
            return res.get("records", [])

        return _make_sync_async_callable(_run)()

    def vfs_search(self, query: str, limit: int = 10, enable_rag: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """Executes hybrid dense vector + metadata search across files."""
        async def _run():
            res = await self.client.vfs_search(query=query, limit=limit, enable_rag=enable_rag)
            return res.get("results", [])

        return _make_sync_async_callable(_run)()


class RemoteSchemaContext(RemoteBaseContext):
    """Context for schema inspection and discovery."""

    def get(self, api_name: str) -> Dict[str, Any]:
        """Get the schema definition for a specific object."""
        coro = self.client.schema_get(api_name=api_name)
        return _make_sync_async_callable(lambda: coro)()

    def list(self) -> List[Dict[str, Any]]:
        """List all schemas available for the organization."""
        async def _run():
            res = await self.client.schema_get()
            if isinstance(res, dict):
                return [
                    {"api_name": k, "title": v.get("title") or v.get("name"), "prefix": v.get("prefix")}
                    for k, v in res.items()
                    if isinstance(v, dict)
                ]
            return res if isinstance(res, list) else []

        return _make_sync_async_callable(_run)()

    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new custom object schema."""
        coro = self.client.schema_create(data=data)
        return _make_sync_async_callable(lambda: coro)()

    def update(self, data: Dict[str, Any]) -> Any:
        """Update an existing object schema."""
        coro = self.client.schema_update(data=data)
        return _make_sync_async_callable(lambda: coro)()

    def create_field(self, data: Dict[str, Any]) -> Any:
        """Create a field in an object schema."""
        coro = self.client.schema_create_field(data=data)
        return _make_sync_async_callable(lambda: coro)()

    def delete(self, schema_id: str) -> Any:
        """Delete a custom object schema."""
        coro = self.client.schema_delete(schema_id=schema_id)
        return _make_sync_async_callable(lambda: coro)()


class RemoteFileContext(RemoteBaseContext):
    """Context for Virtual File Service (VFS) operations."""

    def get_file(self, file_id: str) -> Dict[str, Any]:
        """Loads full file metadata and inline text content or presigned URL in a single roundtrip."""
        coro = self.client.vfs_get_file(file_id=file_id)
        return _make_sync_async_callable(lambda: coro)()

    def search(self, query: str, limit: int = 10, enable_rag: bool = False) -> List[Dict[str, Any]]:
        """Searches files across the organization vault."""
        return self.platform.query.vfs_search(query=query, limit=limit, enable_rag=enable_rag)

    def browse_vault(self, vault_id: str = "root", bypass_cache: bool = False) -> Dict[str, Any]:
        """Inspects folder contents in a vault."""
        coro = self.client.vfs_browse_vault(vault_id=vault_id, bypass_cache=bypass_cache)
        return _make_sync_async_callable(lambda: coro)()

    def browse_path(self, path: str) -> Dict[str, Any]:
        """Resolves human path (e.g. 'Finance/2026') to folder items."""
        coro = self.client.vfs_browse_path(string_path=path)
        return _make_sync_async_callable(lambda: coro)()

    def get_tree(self, bypass_cache: bool = False) -> Dict[str, Any]:
        """Fetches full vault directory tree."""
        coro = self.client.vfs_get_tree(bypass_cache=bypass_cache)
        return _make_sync_async_callable(lambda: coro)()

    def get_snapshot(self) -> Dict[str, Any]:
        """Fetches full hierarchy snapshot of vaults and files."""
        coro = self.client.vfs_get_snapshot()
        return _make_sync_async_callable(lambda: coro)()

    def move_item(self, item_id: str, to_vault_id: Optional[str] = None, from_vault_id: Optional[str] = None) -> Dict[str, Any]:
        """Moves a file or vault to another vault."""
        coro = self.client.vfs_move_item(item_id=item_id, to_vault_id=to_vault_id, from_vault_id=from_vault_id)
        return _make_sync_async_callable(lambda: coro)()

    def delete(self, item_id: str) -> Dict[str, Any]:
        """Deletes a file or vault from VFS."""
        coro = self.client.vfs_delete_item(item_id=item_id)
        return _make_sync_async_callable(lambda: coro)()


class RemoteSlackContext(RemoteBaseContext):
    """Context for Slack operations matching server SlackContext signatures."""

    def post_message(
        self,
        channel: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        as_user: bool = False,
        mrkdwn: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Sends a message or Block Kit payload to a Slack channel or thread."""
        coro = self.client.slack_post_message(
            channel=channel,
            text=text,
            blocks=blocks,
            thread_ts=thread_ts,
            as_user=as_user,
            mrkdwn=mrkdwn,
        )
        return _make_sync_async_callable(lambda: coro)()

    def update_message(
        self,
        channel: str,
        ts: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Updates an existing message in a Slack channel."""
        coro = self.client.slack_update_message(channel=channel, ts=ts, text=text, blocks=blocks)
        return _make_sync_async_callable(lambda: coro)()

    def delete_message(self, channel: str, ts: str, as_user: bool = False) -> Dict[str, Any]:
        """Deletes a message from a Slack channel."""
        coro = self.client.slack_delete_message(channel=channel, ts=ts, as_user=as_user)
        return _make_sync_async_callable(lambda: coro)()

    def list_channels(
        self,
        types: str = "public_channel,private_channel",
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Lists channels in the connected Slack workspace."""
        coro = self.client.slack_list_channels(types=types, cursor=cursor, limit=limit)
        return _make_sync_async_callable(lambda: coro)()

    def get_channel_history(
        self,
        channel: str,
        limit: int = 50,
        latest: Optional[str] = None,
        oldest: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves recent message history or thread conversations from a channel."""
        coro = self.client.slack_get_channel_history(
            channel=channel, limit=limit, latest=latest, oldest=oldest, cursor=cursor
        )
        return _make_sync_async_callable(lambda: coro)()

    def list_users(self, cursor: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Lists members in the workspace."""
        coro = self.client.slack_list_users(cursor=cursor, limit=limit)
        return _make_sync_async_callable(lambda: coro)()

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieves profile for a specific Slack user ID."""
        coro = self.client.slack_get_user_profile(user_id=user_id)
        return _make_sync_async_callable(lambda: coro)()

    def add_reaction(self, channel: str, timestamp: str, name: str) -> Dict[str, Any]:
        """Adds an emoji reaction to a message."""
        clean_name = name.strip().strip(":")
        coro = self.client.slack_add_reaction(channel=channel, timestamp=timestamp, name=clean_name)
        return _make_sync_async_callable(lambda: coro)()

    def get_auth_status(self) -> Dict[str, Any]:
        """Returns Slack integration connection status and workspace metadata."""
        coro = self.client.slack_get_auth_status()
        return _make_sync_async_callable(lambda: coro)()


class RemoteIntegrationContext:
    """Context container for third-party integrations (Slack, Microsoft, Stripe, Scraper)."""

    def __init__(self, platform: 'RemotePlatformContext'):
        self.slack = RemoteSlackContext(platform)
        self.scraper = RemoteScraperContext(platform)
        self.web = self.scraper


class RemoteScraperContext(RemoteBaseContext):
    """Client-Side Context for Web Scraping, Search, Crawling, and Content Diff."""

    def __init__(self, platform: 'RemotePlatformContext'):
        super().__init__(platform)
        from tools.web_tools import WebScraperClient
        self.web_client = WebScraperClient()

    def scrape(self, url: str, force_refresh: bool = False, include_html: bool = False, **kwargs) -> Dict[str, Any]:
        """Scrapes or retrieves clean AI-ready Markdown for any webpage."""
        coro = self.web_client.scrape(url=url, force_refresh=force_refresh, include_html=include_html)
        return _make_sync_async_callable(lambda: coro)()

    def search(self, query: str, limit: int = 5, auto_scrape_top: int = 0, **kwargs) -> Dict[str, Any]:
        """Searches the live public web and returns snippets and URLs."""
        coro = self.web_client.search(query=query, limit=limit, auto_scrape_top=auto_scrape_top)
        return _make_sync_async_callable(lambda: coro)()

    def crawl(self, url: str, max_pages: int = 10, max_concurrency: int = 5, force_refresh: bool = False, **kwargs) -> Dict[str, Any]:
        """Recursively crawls internal pages across a company domain."""
        coro = self.web_client.crawl(url=url, max_pages=max_pages, max_concurrency=max_concurrency, force_refresh=force_refresh)
        return _make_sync_async_callable(lambda: coro)()

    def diff(self, url: str, force_live_check: bool = True, **kwargs) -> Dict[str, Any]:
        """Compares live webpage content against the permanent archive."""
        coro = self.web_client.diff(url=url, force_live_check=force_live_check)
        return _make_sync_async_callable(lambda: coro)()

    def poll_feed(self, feed_url: str, auto_scrape_articles: bool = True, max_articles: int = 10, **kwargs) -> Dict[str, Any]:
        """Polls an RSS/Atom news feed and extracts articles."""
        coro = self.web_client.poll_feed(feed_url=feed_url, auto_scrape_articles=auto_scrape_articles, max_articles=max_articles)
        return _make_sync_async_callable(lambda: coro)()


class RemotePlatformContext:
    """Client-Side Platform Context matching server PlatformContext stubs.
    
    Provides unified access to:
      - platform.records (create, update, delete, hydrate_batch, merge)
      - platform.query (sql, mongo, vfs_search)
      - platform.schema (get, list, create, update, create_field, delete)
      - platform.files (get_file, search, browse_vault, browse_path, get_tree, get_snapshot, move_item, delete)
      - platform.slack / platform.integrations.slack (post_message, list_channels, get_channel_history, list_users, etc.)
      - platform.scraper / platform.web (scrape, search, crawl, diff, poll_feed)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[ValstormApiClient] = None,
    ):
        if client is not None:
            self.client = client
        else:
            self.client = ValstormApiClient(token=token, base_url=base_url)

        self.records = RemoteRecordContext(self)
        self.query = RemoteQueryContext(self)
        self.schema = RemoteSchemaContext(self)
        self.files = RemoteFileContext(self)
        self.slack = RemoteSlackContext(self)
        self.scraper = RemoteScraperContext(self)
        self.web = self.scraper
        self.integrations = RemoteIntegrationContext(self)

    def __repr__(self) -> str:
        return f"<RemotePlatformContext base_url='{self.client.base_url}' authenticated={bool(self.client.token)}>"
