"""Main entry point and Typer application for the Valstorm Agent Runtime (vsagent)."""

import sys
from typing import Optional
import typer
from rich.console import Console

from .cmds.chat_cmds import chat_command, run_command
from .cmds.doctor_cmd import doctor_command
from .cmds.keys_cmds import keys_app
from .cmds.memory_cmds import memory_app
from .cmds.models_cmd import models_command
from .cmds.profiles_cmds import profiles_app
from .cmds.server_cmds import server_app
from .cmds.sessions_cmds import sessions_app
from .cmds.skills_cmds import skills_app
from .cmds.sync_cmds import catalog_app
from .cmds.status_cmd import status_command, update_command, version_command, version_callback

console = Console()

app = typer.Typer(
    name="vsagent",
    help="Valstorm Autonomous AI Agent Runtime & Toolbelt CLI",
    no_args_is_help=False,
)

# Sub-command groups
app.add_typer(profiles_app, name="profiles")
app.add_typer(skills_app, name="skills")
app.add_typer(catalog_app, name="catalog")
app.add_typer(catalog_app, name="sync", hidden=True)
app.add_typer(sessions_app, name="sessions")
app.add_typer(memory_app, name="memory")
app.add_typer(keys_app, name="keys")
app.add_typer(server_app, name="server")

# Primary execution commands
app.command(name="chat", help="Start an interactive multi-turn AI chat REPL")(chat_command)
app.command(name="run", help="Execute a single task prompt")(run_command)
app.command(name="status", help="Show system status, provider keys, and active workspace")(status_command)
app.command(name="doctor", help="Check every failover tier: key source, model IDs and a live probe")(doctor_command)
app.command(name="version", help="Show the vsagent CLI version")(version_command)
app.command(name="models", help="List models the Valstorm gateway accepts and what they resolve to")(models_command)
app.command(name="update", help="Update the vsagent CLI to latest version")(update_command)


@app.callback(invoke_without_command=True)
def default_entrypoint(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        "-V",
        help="Show vsagent version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Task prompt to execute immediately"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Agent profile (e.g. developer, researcher, orchestrator)"),
    model: str = typer.Option("gemini-flash-latest", "--model", "-m", help="Model name"),
    provider: str = typer.Option("valstorm", "--provider", help="Provider (valstorm, gemini, openai, anthropic)"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume session ID"),
    env: str = typer.Option("prod", "--env", "-e", help="Valstorm environment (prod, dev, local)"),
):
    """
    Valstorm Autonomous AI Agent Runtime & CLI (vsagent).
    
    If invoked without a command, launches interactive chat REPL.
    """
    if ctx.invoked_subcommand is None:
        if prompt:
            run_command(
                prompt=prompt,
                profile=profile,
                model=model,
                provider=provider,
                api_key=None,
                env=env,
                token=None,
                resume=resume,
                max_iterations=None,
            )
        else:
            chat_command(
                profile=profile,
                model=model,
                provider=provider,
                api_key=None,
                env=env,
                token=None,
                resume=resume,
                max_iterations=None,
            )


def main():
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting...[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
