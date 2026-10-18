"""Benchmark Adapters Package."""

from .base import BenchmarkAdapter, BenchmarkConfig, ExecutionTrace, estimate_cost_usd
from .external_cli_adapter import (
    HARNESS_BINARY_MAP,
    ExternalCliBenchmarkAdapter,
    build_cli_command,
    is_harness_installed,
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
    "VsAgentBenchmarkAdapter",
    "TaskAwareMockProvider",
    "build_benchmark_tool_registry",
    "ExternalCliBenchmarkAdapter",
    "HARNESS_BINARY_MAP",
    "build_cli_command",
    "is_harness_installed",
]
