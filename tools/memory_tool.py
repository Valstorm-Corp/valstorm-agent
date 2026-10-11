"""Memory Management and Session Search Tools for Agent Runtime."""

import asyncio
import json
import os
from typing import Any, Dict, List, Literal, Optional
import httpx

from core.memory import MemoryStore
from core.storage import SessionStore
from core.tools import ToolRegistry, tool
from tools.valstorm_client import resolve_valstorm_credentials


def create_memory_tools(
    memory_store: Optional[MemoryStore] = None,
    session_store: Optional[SessionStore] = None,
) -> List[Any]:
    mem_store = memory_store or MemoryStore()
    sess_store = session_store or SessionStore()

    @tool
    async def memory_manage(
        action: Literal["add", "remove", "list"],
        target: Literal["memory", "user"] = "memory",
        content: Optional[str] = None,
        old_text: Optional[str] = None,
    ) -> str:
        """Manages persistent declarative memory facts (user profile or environment notes).

        Args:
            action: 'add' to save a new fact, 'remove' to delete a fact, or 'list' to view stored facts.
            target: 'user' for facts about the user/preferences, 'memory' for environment/codebase notes.
            content: The text of the fact to add (required when action='add').
            old_text: Substring text identifying the fact to remove (required when action='remove').
        """
        token, base_url = resolve_valstorm_credentials()
        # If in cloud/remote mode and managing user memory, route mutations to tenant FastAPI
        if token and base_url and target == "user":
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            client_timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=client_timeout) as http_client:
                try:
                    if action == "add":
                        if not content:
                            return "Error: `content` is required when action='add'."
                        resp = await http_client.post(
                            f"{base_url}/ai/memory/fact",
                            headers=headers,
                            json={"fact": content, "category": "user_fact"},
                        )
                        if resp.status_code == 200:
                            return f"Saved fact to tenant memory: {content}"
                    elif action == "remove":
                        if not old_text:
                            return "Error: `old_text` is required when action='remove'."
                        req = http_client.build_request(
                            "DELETE",
                            f"{base_url}/ai/memory/fact",
                            headers=headers,
                            json={"old_text": old_text},
                        )
                        resp = await http_client.send(req)
                        if resp.status_code == 200:
                            return f"Removed fact matching '{old_text}' from tenant memory."
                    elif action == "list":
                        resp = await http_client.get(
                            f"{base_url}/ai/memory/working-context",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            return json.dumps(data.get("declarative_facts", []), indent=2)
                except Exception as e:
                    # Fallback to local SQLite store on network failure
                    pass

        # Default local SQLite store handling
        if action == "list":
            facts = mem_store.get_facts(target=target)
            return json.dumps(facts, indent=2)

        elif action == "add":
            if not content:
                return "Error: `content` is required when action='add'."
            return mem_store.add_fact(target=target, content=content)

        elif action == "remove":
            if not old_text:
                return "Error: `old_text` is required when action='remove'."
            removed = mem_store.remove_fact(target=target, old_text=old_text)
            if removed:
                return f"Successfully removed fact matching '{old_text}' from [{target}] store."
            return f"No fact found matching '{old_text}' in [{target}] store."

        return f"Error: Unknown action '{action}'. Choose 'add', 'remove', or 'list'."

    @tool
    async def session_search(query: str, limit: int = 5) -> str:
        """Searches past conversation sessions and turns using full-text search (FTS5).

        Args:
            query: Keywords or phrases to search in past sessions.
            limit: Maximum number of search results to return (default: 5).
        """
        results = await asyncio.to_thread(sess_store.search_sessions, query=query, limit=limit)
        if not results:
            return f"No past sessions found matching '{query}'."

        formatted = []
        for r in results:
            formatted.append(
                f"- [{r['title']}] (Session: {r['session_id'][:8]} · {r['role'].capitalize()} · {r['updated_at'][:10]}):\n"
                f"  Snippet: {r['snippet']}"
            )
        return "\n".join(formatted)

    return [memory_manage, session_search]


def register_memory_tools(
    registry: ToolRegistry,
    memory_store: Optional[MemoryStore] = None,
    session_store: Optional[SessionStore] = None,
) -> ToolRegistry:
    """Registers memory_manage and session_search tools into the ToolRegistry."""
    tools = create_memory_tools(memory_store=memory_store, session_store=session_store)
    for t in tools:
        registry.register(t)
    return registry
