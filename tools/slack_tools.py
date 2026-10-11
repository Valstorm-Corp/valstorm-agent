"""Slack Integration tools for Valstorm Agent Runtime.

Provides tools for messaging, channel listing, message history, user lookup,
reactions, message updates, and status checking via the connected Valstorm Slack integration.
"""

import json
from typing import Any, Dict, List, Optional, Union

from core.tools import ToolRegistry, tool
from tools.valstorm_client import ValstormApiClient


def create_slack_tools(client: Optional[ValstormApiClient] = None) -> List[Any]:
    """Creates Slack tool functions bound to a specific ValstormApiClient instance."""
    api_client = client or ValstormApiClient()

    @tool
    async def slack_post_message(
        channel: str,
        text: Optional[str] = None,
        blocks: Optional[Union[str, List[Dict[str, Any]]]] = None,
        thread_ts: Optional[str] = None,
        as_user: bool = False,
        mrkdwn: bool = True,
    ) -> str:
        """Sends a message or Block Kit payload to a Slack channel or thread.

        Args:
            channel: Slack channel ID (e.g. 'C01234567') or channel name. Use slack_list_channels to find channel IDs.
            text: Message text content (supports Slack mrkdwn formatting). Optional if blocks are provided.
            blocks: Block Kit layout blocks (list of block dicts or a JSON string).
            thread_ts: Message timestamp (e.g. '1712345678.123456') to reply within an existing thread.
            as_user: If True, post as the authenticated user rather than the bot.
            mrkdwn: Whether to enable standard Slack markdown formatting (default: True).
        """
        if not channel or not channel.strip():
            return "Error: channel parameter is required."
        if not text and not blocks:
            return "Error: Either 'text' or 'blocks' must be provided to post a Slack message."

        data_blocks = blocks
        if isinstance(blocks, str):
            try:
                data_blocks = json.loads(blocks)
            except Exception as e:
                return f"Error: blocks argument must be valid JSON: {e}"

        res = await api_client.slack_post_message(
            channel=channel.strip(),
            text=text,
            blocks=data_blocks,
            thread_ts=thread_ts.strip() if thread_ts else None,
            as_user=as_user,
            mrkdwn=mrkdwn,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_list_channels(
        types: str = "public_channel,private_channel",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> str:
        """Lists channels in the connected Slack workspace to discover channel names, IDs, and topics.

        Args:
            types: Comma-separated channel types to include (e.g. 'public_channel,private_channel').
            limit: Maximum number of channels to return (default: 100, max: 200).
            cursor: Pagination cursor for fetching subsequent pages.
        """
        res = await api_client.slack_list_channels(
            types=types,
            cursor=cursor,
            limit=limit,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_get_channel_history(
        channel: str,
        limit: int = 50,
        latest: Optional[str] = None,
        oldest: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Retrieves recent message history or thread conversations from a Slack channel.

        Args:
            channel: Slack channel ID (e.g. 'C01234567').
            limit: Number of recent messages to retrieve (default: 50, max: 100).
            latest: End of time range of messages to include (Slack timestamp).
            oldest: Start of time range of messages to include (Slack timestamp).
            cursor: Pagination cursor for next page of messages.
        """
        if not channel or not channel.strip():
            return "Error: channel parameter is required."

        res = await api_client.slack_get_channel_history(
            channel=channel.strip(),
            limit=limit,
            latest=latest,
            oldest=oldest,
            cursor=cursor,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_list_users(
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> str:
        """Lists members of the connected Slack workspace to lookup user IDs, real names, and emails.

        Args:
            limit: Maximum number of users to return (default: 100).
            cursor: Pagination cursor for next page.
        """
        res = await api_client.slack_list_users(cursor=cursor, limit=limit)
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_get_user_profile(user_id: str) -> str:
        """Retrieves detailed profile information (real name, display name, email, title, avatar) for a specific Slack user.

        Args:
            user_id: Slack user ID (e.g. 'U01234567').
        """
        if not user_id or not user_id.strip():
            return "Error: user_id parameter is required."

        res = await api_client.slack_get_user_profile(user_id=user_id.strip())
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_add_reaction(
        channel: str,
        timestamp: str,
        name: str,
    ) -> str:
        """Adds an emoji reaction to a message in a Slack channel.

        Args:
            channel: Slack channel ID where the message is located.
            timestamp: Timestamp of the message to react to (e.g. '1712345678.123456').
            name: Emoji name without colons (e.g. 'white_check_mark', 'eyes', 'thumbsup', 'rocket').
        """
        if not channel or not channel.strip():
            return "Error: channel parameter is required."
        if not timestamp or not timestamp.strip():
            return "Error: timestamp parameter is required."
        if not name or not name.strip():
            return "Error: emoji name parameter is required."

        clean_name = name.strip().strip(":")
        res = await api_client.slack_add_reaction(
            channel=channel.strip(),
            timestamp=timestamp.strip(),
            name=clean_name,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_update_message(
        channel: str,
        ts: str,
        text: Optional[str] = None,
        blocks: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ) -> str:
        """Updates an existing message in a Slack channel.

        Args:
            channel: Slack channel ID containing the message.
            ts: Timestamp of the message to update (e.g. '1712345678.123456').
            text: New text content for the message.
            blocks: New Block Kit layout blocks (list of block dicts or JSON string).
        """
        if not channel or not channel.strip():
            return "Error: channel parameter is required."
        if not ts or not ts.strip():
            return "Error: ts (timestamp) parameter is required."
        if not text and not blocks:
            return "Error: Either 'text' or 'blocks' must be provided to update a Slack message."

        data_blocks = blocks
        if isinstance(blocks, str):
            try:
                data_blocks = json.loads(blocks)
            except Exception as e:
                return f"Error: blocks argument must be valid JSON: {e}"

        res = await api_client.slack_update_message(
            channel=channel.strip(),
            ts=ts.strip(),
            text=text,
            blocks=data_blocks,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_delete_message(
        channel: str,
        ts: str,
        as_user: bool = False,
    ) -> str:
        """Deletes a message from a Slack channel.

        Args:
            channel: Slack channel ID containing the message.
            ts: Timestamp of the message to delete.
            as_user: If True, delete as the authenticated user.
        """
        if not channel or not channel.strip():
            return "Error: channel parameter is required."
        if not ts or not ts.strip():
            return "Error: ts (timestamp) parameter is required."

        res = await api_client.slack_delete_message(
            channel=channel.strip(),
            ts=ts.strip(),
            as_user=as_user,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def slack_get_auth_status() -> str:
        """Checks current Slack integration connection status, team name, team ID, bot user ID, and active scopes."""
        res = await api_client.slack_get_auth_status()
        return json.dumps(res, indent=2, default=str)

    return [
        slack_post_message,
        slack_list_channels,
        slack_get_channel_history,
        slack_list_users,
        slack_get_user_profile,
        slack_add_reaction,
        slack_update_message,
        slack_delete_message,
        slack_get_auth_status,
    ]


def register_slack_tools(
    registry: ToolRegistry,
    client: Optional[ValstormApiClient] = None,
    token: Optional[str] = None,
    env: str = "local",
) -> ToolRegistry:
    """Registers all Slack tools into a ToolRegistry."""
    api_client = client or ValstormApiClient(token=token, env=env)
    tools = create_slack_tools(client=api_client)
    for t in tools:
        registry.register(t)
    return registry
