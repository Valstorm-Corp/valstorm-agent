#!/usr/bin/env python3
"""vsagent ToolRegistry -> MCP stdio bridge.

Serves the exact same tool implementations `vsagent` uses to any MCP client (Claude Code,
Codex, OpenCode, ...), so head-to-head benchmarks hold the tool surface constant and only
vary the harness and model.

Tools that duplicate the client's built-ins (file I/O, shell, web fetch, subagents) or that
mutate host-global vsagent state (memory, skill authoring) are excluded by default.

Usage (Claude Code):
    claude -p "<goal>" --strict-mcp-config --mcp-config '{"mcpServers": {"vsagent": {
        "type": "stdio", "command": "<agent-runtime>/.venv/bin/python",
        "args": ["<agent-runtime>/benchmarks/mcp_bridge.py", "--workspace", "<dir>"]}}}'

Backends:
    --backend sandbox  (default) Valstorm tools run against BenchmarkValstormClient in --workspace
    --backend live     Valstorm tools run against the real platform with your local credentials
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

RUNTIME_ROOT = Path(__file__).resolve().parent.parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

# MCP owns stdout; anything a tool prints must go to stderr instead.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

from core.sandbox import HostSandbox, set_current_sandbox  # noqa: E402
from core.tools import ToolRegistry  # noqa: E402

SERVER_NAME = "vsagent"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

# Duplicates of MCP-client built-ins, or tools that write host-global vsagent state.
BRIDGE_EXCLUDED_TOOLS = frozenset({
    "read_file",
    "read_local_file",
    "write_file",
    "patch_file",
    "terminal_exec",
    "search_files",
    "execute_code",
    "process_manage",
    "web_search",
    "web_scrape",
    "delegate_task",
    "subagent_manage",
    "agent_manage",
    "clarify",
    "confirmation_required",
    "memory_manage",
    "session_search",
    "skill_manage",
    "mock_db_lookup",
})


def build_bridge_registry(
    workspace_dir: Path,
    tools_mode: str = "full",
    whitelist: Optional[List[str]] = None,
    backend: str = "sandbox",
) -> ToolRegistry:
    """Builds the vsagent ToolRegistry to expose, minus client built-in duplicates."""
    from benchmarks.adapters.vsagent_adapter import _EXCLUDED_EXTERNAL_TOOLS, build_benchmark_tool_registry

    task: Dict[str, Any] = {"tool_chain": list(whitelist or [])}
    effective_mode = "scoped" if (tools_mode == "scoped" and whitelist) else "full"
    if backend == "live":
        from tools.valstorm_tools import create_valstorm_tools

        registry = build_benchmark_tool_registry(effective_mode, task, workspace_dir=None)
        live_tools = [vt for vt in create_valstorm_tools() if getattr(vt, "_tool_name", "") not in _EXCLUDED_EXTERNAL_TOOLS]
        for vt in live_tools:
            if effective_mode == "full" or getattr(vt, "_tool_name", "") in task["tool_chain"]:
                registry.register(vt)
    else:
        registry = build_benchmark_tool_registry(effective_mode, task, workspace_dir=workspace_dir)

    exposed = [name for name in registry.list_tools() if name not in BRIDGE_EXCLUDED_TOOLS]
    return registry.filter_by_whitelist(exposed)


def tool_definitions(registry: ToolRegistry) -> List[Dict[str, Any]]:
    """Converts vsagent tool schemas into MCP `tools/list` entries."""
    defs = []
    for schema in registry.get_schemas():
        input_schema = dict(schema.get("parameters") or {})
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        defs.append({
            "name": schema["name"],
            "description": schema.get("description", ""),
            "inputSchema": input_schema,
        })
    return defs


class McpBridgeServer:
    """Minimal MCP (JSON-RPC 2.0 over newline-delimited stdio) server for a ToolRegistry."""

    def __init__(self, registry: ToolRegistry, trace_file: Optional[Path] = None):
        self.registry = registry
        self.trace_file = trace_file

    def _trace(self, record: Dict[str, Any]) -> None:
        if self.trace_file:
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    async def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method")
        msg_id = msg.get("id")
        if msg_id is None:
            return None  # notification (e.g. notifications/initialized)

        if method == "initialize":
            requested = (msg.get("params") or {}).get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
            result: Dict[str, Any] = {
                "protocolVersion": requested,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_definitions(self.registry)}
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            t0 = time.perf_counter()
            res = await self.registry.execute_async(name, args)
            self._trace({
                "tool": name,
                "arguments": args,
                "is_error": res.is_error,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            })
            result = {"content": [{"type": "text", "text": res.output or ""}], "isError": bool(res.is_error)}
        else:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    async def serve(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            try:
                response = await self.handle(msg)
            except Exception as e:
                response = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32603, "message": str(e)}}
            if response is not None:
                self._write(response)

    @staticmethod
    def _write(payload: Dict[str, Any]) -> None:
        _PROTOCOL_OUT.write(json.dumps(payload) + "\n")
        _PROTOCOL_OUT.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the vsagent ToolRegistry over MCP stdio")
    parser.add_argument("--workspace", required=True, help="Workspace directory tools operate in")
    parser.add_argument("--tools-mode", choices=["none", "scoped", "full"], default="full")
    parser.add_argument("--tools", default="", help="Comma-separated whitelist (with --tools-mode scoped)")
    parser.add_argument("--backend", choices=["sandbox", "live"], default="sandbox")
    parser.add_argument("--trace-file", default=None, help="Optional JSONL log of every tool call")
    args = parser.parse_args()

    workspace_dir = Path(args.workspace).resolve()
    set_current_sandbox(HostSandbox(base_dir=str(workspace_dir)))

    # Match the vsagent benchmark adapter: no nested vision-LLM calls skewing the measurement.
    if args.backend == "sandbox":
        from benchmarks.adapters.vsagent_adapter import _BenchmarkVisionInspector
        from tools.vision_tool import set_vision_provider_override

        set_vision_provider_override(_BenchmarkVisionInspector())

    if args.tools_mode == "none":
        registry = ToolRegistry()
    else:
        whitelist = [t.strip() for t in args.tools.split(",") if t.strip()]
        registry = build_bridge_registry(workspace_dir, args.tools_mode, whitelist, args.backend)

    server = McpBridgeServer(registry, Path(args.trace_file) if args.trace_file else None)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
