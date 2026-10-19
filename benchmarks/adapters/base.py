"""Base adapter interfaces and configuration models for the Agent Benchmark Harness."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# Rough per-1M token pricing table ($ USD) for cost estimation across providers
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gemini-flash-latest": {"input": 0.15, "output": 0.60},
    "gemini-flash-lite-latest": {"input": 0.075, "output": 0.30},
    "gemini-pro-latest": {"input": 1.25, "output": 5.00},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "openai-gpt-oss-120b": {"input": 0.20, "output": 0.60},
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimates USD cost for a benchmark run based on model token pricing (with 75% discount on cached prefix tokens)."""
    rates = MODEL_PRICING_PER_1M.get(model, {"input": 0.50, "output": 1.50})
    safe_cached = max(0, min(prompt_tokens, int(cached_tokens or 0)))
    uncached_prompt = max(0, prompt_tokens - safe_cached)
    input_cost = ((uncached_prompt * rates["input"]) + (safe_cached * rates["input"] * 0.25)) / 1_000_000.0
    output_cost = (completion_tokens / 1_000_000.0) * rates["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class BenchmarkConfig:
    """Defines a single point in the 10-factor agentic configuration ablation space.

    Covers:
    1-2. With skills (`with_skills=True`) vs. Without skills (`with_skills=False`)
    3-4. With ReAct loop (`with_react_loop=True`) vs. Without ReAct loop (`with_react_loop=False`, single-shot)
    5-6. With Tools (`tools_mode='scoped'` or `'full'`) vs. Without Tools (`tools_mode='none'`)
    7.   Model & Reasoning / Thinking Level (`provider`, `model`, `thinking_level`)
    9.   Agent Harness (`harness`: 'vsagent', 'claude', 'codex', 'opencode', 'hermes', 'oh-my-pi')
    10.  Swarm / Parallel Performance (`swarm_mode=True/False`, `swarm_workers`)
    """

    config_id: str
    label: str
    harness: Literal["vsagent", "claude", "codex", "opencode", "hermes", "oh-my-pi"] = "vsagent"
    provider: str = "gemini"
    model: str = "gemini-flash-latest"
    thinking_level: Literal["none", "low", "medium", "high", "adaptive"] = "medium"
    with_skills: bool = True
    skills: List[str] = field(default_factory=list)
    with_react_loop: bool = True
    max_turns: int = 15
    tools_mode: Literal["none", "scoped", "full"] = "scoped"
    swarm_mode: bool = False
    swarm_workers: int = 1
    use_decision_model: bool = True
    auto_tune: bool = False
    timeout_sec: float = 120.0

    @property
    def effective_max_turns(self) -> int:
        """Single-shot (1 turn) when ReAct loop is disabled; otherwise max_turns."""
        return self.max_turns if self.with_react_loop else 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    """Captures telemetry and trajectory details from a single benchmark task execution."""

    task_id: str
    config_id: str
    harness: str
    status: Literal["completed", "failed", "timed_out", "skipped"]
    final_output: str = ""
    tool_calls_made: List[str] = field(default_factory=list)
    tool_outputs: List[str] = field(default_factory=list)
    turns_taken: int = 0
    duration_sec: float = 0.0
    llm_time_sec: float = 0.0
    tool_time_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    subagents_spawned: int = 0
    sequential_sum_sec: float = 0.0
    parallel_speedup: float = 1.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkAdapter(ABC):
    """Abstract base class for running a benchmark task inside an isolated workspace."""

    @abstractmethod
    async def execute_task(
        self,
        task: Dict[str, Any],
        config: BenchmarkConfig,
        workspace_dir: Path,
        mode: Literal["auto", "mock", "live"] = "auto",
    ) -> ExecutionTrace:
        """Runs `task` under `config` inside `workspace_dir` and returns the execution trace."""
        pass
