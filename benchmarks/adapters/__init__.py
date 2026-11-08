"""Benchmark Adapters Package."""

from .base import BenchmarkAdapter, BenchmarkConfig, ExecutionTrace, detect_grader_tampering, estimate_cost_usd
from .external_cli_adapter import (
    HARNESS_BINARY_MAP,
    ExternalCliBenchmarkAdapter,
    build_claude_env,
    build_cli_command,
    build_mcp_bridge_config,
    is_harness_installed,
    parse_claude_stream,
)
from .vsagent_adapter import (
    TaskAwareMockProvider,
    VsAgentBenchmarkAdapter,
    build_benchmark_tool_registry,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkConfig",
    "ExecutionTrace",
    "estimate_cost_usd",
    "detect_grader_tampering",
    "VsAgentBenchmarkAdapter",
    "TaskAwareMockProvider",
    "build_benchmark_tool_registry",
    "ExternalCliBenchmarkAdapter",
    "HARNESS_BINARY_MAP",
    "build_cli_command",
    "build_claude_env",
    "build_mcp_bridge_config",
    "parse_claude_stream",
    "is_harness_installed",
]
