"""Server management commands for Valstorm Agent HTTP/SSE Gateway."""

import os
import subprocess
import sys
from typing import Optional
import httpx
import typer
from rich.console import Console

server_app = typer.Typer(help="Manage the local Agent Runtime API & SSE server", no_args_is_help=False)
console = Console()


@server_app.command(name="start")
def start_server(
    mode_arg: Optional[str] = typer.Argument(
        None,
        metavar="[host|cloud]",
        help="Runtime mode: 'host' (port 8650) or 'cloud' (port 8660). E.g. 'vsagent server start host' or 'vsagent server start cloud'",
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Server port (defaults to 8650 for host, 8660 for cloud)"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Runtime mode: 'host' or 'cloud'"),
    default_sandbox: Optional[str] = typer.Option(None, "--default-sandbox", "-s", help="Default sandbox: 'host', 'docker', or 'cloud'"),
    dev: bool = typer.Option(False, "--dev", "-d", help="Run in dev mode with live debug logging and auto-reload"),
    log_file: Optional[str] = typer.Option(None, "--log-file", "-l", help="Path to write server execution logs"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the Valstorm Agent Runtime FastAPI HTTP / SSE server."""
    import uvicorn
    from server import setup_server_logging

    raw_mode_arg = None if hasattr(mode_arg, "default") else mode_arg
    raw_mode = None if hasattr(mode, "default") else mode
    raw_sandbox = None if hasattr(default_sandbox, "default") else default_sandbox
    raw_port = None if hasattr(port, "default") else port
    raw_host = "0.0.0.0" if hasattr(host, "default") else host
    raw_dev = False if hasattr(dev, "default") else bool(dev)
    raw_reload = False if hasattr(reload, "default") else bool(reload)
    raw_log_file = None if hasattr(log_file, "default") else log_file

    should_reload = raw_reload or raw_dev
    log_level = "DEBUG" if raw_dev else "INFO"

    chosen_mode = (raw_mode_arg or raw_mode or "host").lower().strip()
    chosen_sandbox = (raw_sandbox or ("cloud" if chosen_mode == "cloud" else "host")).lower().strip()
    chosen_port = raw_port if raw_port is not None else (8660 if chosen_mode == "cloud" else 8650)

    os.environ["VALSTORM_AGENT_PORT"] = str(chosen_port)
    os.environ["VALSTORM_AGENT_MODE"] = chosen_mode
    os.environ["VALSTORM_DEFAULT_SANDBOX"] = chosen_sandbox
    os.environ["VALSTORM_LOG_LEVEL"] = log_level

    actual_log_path = setup_server_logging(
        log_level_str=log_level,
        log_file_path=raw_log_file,
        mode=chosen_mode,
        port=chosen_port,
    )

    console.print(
        f"[bold cyan]⚡ Starting Valstorm Agent Gateway on {raw_host}:{chosen_port}[/bold cyan] "
        f"(Mode: [bold]{chosen_mode}[/bold], Default Sandbox: [bold]{chosen_sandbox}[/bold], Dev: [bold]{raw_dev}[/bold])"
    )
    console.print(f"[bold green]✓ Logging active:[/bold green] [cyan]{actual_log_path}[/cyan]\n")

    uvicorn_log_level = "debug" if raw_dev else "info"
    uvicorn.run("server:app", host=raw_host, port=chosen_port, reload=should_reload, log_level=uvicorn_log_level)


@server_app.command(name="status")
def server_status(
    port: int = typer.Option(8650, "--port", "-p", help="Server port to check"),
):
    """Check if local Agent Runtime server is running and responding."""
    url = f"http://localhost:{port}/health"
    try:
        res = httpx.get(url, timeout=3.0)
        if res.status_code == 200:
            console.print(f"[bold green]✓ Server is RUNNING on port {port}[/bold green]")
            console.print(res.json())
        else:
            console.print(f"[yellow]! Server responded with HTTP {res.status_code}[/yellow]")
    except Exception as e:
        console.print(f"[red]✖ Server is NOT running on port {port} ({e})[/red]")
