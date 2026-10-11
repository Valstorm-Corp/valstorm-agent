"""Chat and single-shot task runner commands."""

import asyncio
from typing import Optional
import typer

from ..helpers import run_single_prompt
from ..repl import run_interactive_repl


def run_command(
    prompt: str = typer.Argument(..., help="The prompt or task instruction to execute"),
    profile: Optional[str] = typer.Option("software-engineer", "--profile", "-p", help="Agent profile (e.g. developer, researcher, orchestrator)"),
    model: str = typer.Option("gemini-flash-latest", "--model", "-m", help="Model name (default: gemini-flash-latest)"),
    provider: str = typer.Option("valstorm", "--provider", help="Provider (valstorm, gemini, openai, anthropic)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Explicit API key override"),
    env: str = typer.Option("prod", "--env", "-e", help="Valstorm environment target (local, dev, prod)"),
    token: Optional[str] = typer.Option(None, "--token", help="Valstorm API token override"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", "--session", "-s", help="Resume past session ID"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Maximum ReAct iterations per turn"),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Synchronize sessions with Valstorm Cloud (default: True)"),
):
    """Run a single-prompt autonomous AI task and exit."""
    try:
        asyncio.run(
            run_single_prompt(
                prompt=prompt,
                model=model,
                provider_name=provider,
                api_key=api_key,
                valstorm_env=env,
                valstorm_token=token,
                resume_session_id=resume,
                max_iterations=max_iterations,
                profile=profile,
                cloud_sync=sync,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n\033[91m✖ [Error]\033[0m {e}")


def chat_command(
    profile: Optional[str] = typer.Option("software-engineer", "--profile", "-p", help="Initial agent profile (e.g. developer, researcher, orchestrator)"),
    model: str = typer.Option("gemini-flash-latest", "--model", "-m", help="Initial model name (default: gemini-flash-latest)"),
    provider: str = typer.Option("valstorm", "--provider", help="Provider (valstorm, gemini, openai, anthropic)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Explicit API key override"),
    env: str = typer.Option("prod", "--env", "-e", help="Valstorm environment target (local, dev, prod)"),
    token: Optional[str] = typer.Option(None, "--token", help="Valstorm API token override"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", "--session", "-s", help="Resume past session ID"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Maximum ReAct iterations per turn"),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Synchronize sessions with Valstorm Cloud (default: True)"),
):
    """Start an interactive multi-turn AI Agent chat REPL."""
    try:
        asyncio.run(
            run_interactive_repl(
                initial_model=model,
                initial_provider=provider,
                api_key=api_key,
                valstorm_env=env,
                valstorm_token=token,
                resume_session_id=resume,
                max_iterations=max_iterations,
                profile=profile,
                cloud_sync=sync,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n\033[91m✖ [Error]\033[0m {e}")
