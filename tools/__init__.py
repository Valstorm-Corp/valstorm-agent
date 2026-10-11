"""Valstorm Platform, Developer, Tier 1 Orchestration, and Memory tools package."""

from typing import Any, List

from core.tools import ToolRegistry
from tools.clarify import (
    clarify,
    create_clarify_tools,
    register_clarify_tools,
)
from tools.confirmation import (
    confirmation_required,
    create_confirmation_tools,
    register_confirmation_tools,
)
from tools.delegation import (
    delegate_task,
    subagent_manage,
    create_delegation_tools,
    register_delegation_tools,
)
from tools.developer_tools import (
    create_developer_tools,
    patch_file,
    read_file,
    register_developer_tools,
    search_files,
    terminal_exec,
    write_file,
)
from tools.execute_code import (
    execute_code,
    create_execute_code_tools,
    register_execute_code_tools,
)
from tools.memory_tool import (
    create_memory_tools,
    register_memory_tools,
)
from tools.process_manager import (
    create_process_manager_tools,
    process_manage,
    register_process_manager_tools,
)
from tools.skill_tool import (
    create_skill_tools,
    register_skill_tools,
    skill_list,
    skill_view,
)
from tools.slack_tools import (
    create_slack_tools,
    register_slack_tools,
)
from tools.valstorm_client import (
    ValstormApiClient,
    decode_jwt_payload,
    is_jwt_expired,
    refresh_valstorm_tokens_async,
    refresh_valstorm_tokens_sync,
    resolve_valstorm_auth_context,
    resolve_valstorm_credentials,
)
from tools.valstorm_tools import (
    create_valstorm_tools,
    register_valstorm_tools,
)
from tools.vision_tool import (
    analyze_image,
    create_vision_tools,
    register_vision_tools,
    vision_analyze,
)
from tools.web_tools import (
    WebScraperClient,
    create_web_tools,
    register_web_tools,
    web_content_diff,
    web_crawl_domain,
    web_feed_poll,
    web_scrape,
    web_search,
)


def create_tier1_tools() -> List[Any]:
    """Returns the complete list of Tier 1 orchestration, safety, vision, and web intelligence tools."""
    return [
        confirmation_required,
        clarify,
        analyze_image,
        vision_analyze,
        execute_code,
        delegate_task,
        subagent_manage,
        process_manage,
        skill_view,
        skill_list,
        web_scrape,
        web_search,
        web_crawl_domain,
        web_content_diff,
        web_feed_poll,
    ]


def register_tier1_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers all Tier 1 orchestration and safety tools into the provided ToolRegistry."""
    for tool_fn in create_tier1_tools():
        registry.register(tool_fn)
    return registry


__all__ = [
    "ValstormApiClient",
    "resolve_valstorm_credentials",
    "resolve_valstorm_auth_context",
    "decode_jwt_payload",
    "is_jwt_expired",
    "refresh_valstorm_tokens_sync",
    "refresh_valstorm_tokens_async",
    "create_valstorm_tools",
    "register_valstorm_tools",
    "create_slack_tools",
    "register_slack_tools",
    "terminal_exec",
    "patch_file",
    "write_file",
    "read_file",
    "search_files",
    "create_developer_tools",
    "register_developer_tools",
    "create_memory_tools",
    "register_memory_tools",
    "confirmation_required",
    "create_confirmation_tools",
    "register_confirmation_tools",
    "clarify",
    "create_clarify_tools",
    "register_clarify_tools",
    "execute_code",
    "create_execute_code_tools",
    "register_execute_code_tools",
    "delegate_task",
    "subagent_manage",
    "create_delegation_tools",
    "register_delegation_tools",
    "process_manage",
    "create_process_manager_tools",
    "register_process_manager_tools",
    "skill_view",
    "skill_list",
    "create_skill_tools",
    "register_skill_tools",
    "analyze_image",
    "vision_analyze",
    "create_vision_tools",
    "register_vision_tools",
    "create_tier1_tools",
    "register_tier1_tools",
    "WebScraperClient",
    "create_web_tools",
    "register_web_tools",
    "web_scrape",
    "web_search",
    "web_crawl_domain",
    "web_content_diff",
    "web_feed_poll",
]
