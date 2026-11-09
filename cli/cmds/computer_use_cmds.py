"""CLI subcommands for Computer Use (`vsagent computer-use doctor | install | status`)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import typer
from rich.console import Console
from rich.table import Table

from core.cua_client import get_computer_use_session

console = Console()

computer_use_app = typer.Typer(
    name="computer-use",
    help="Manage and diagnose background Desktop & GUI Computer Use (cua-driver / AX Tree / Set-of-Mark)",
    no_args_is_help=True,
)


@computer_use_app.command(name="doctor")
def computer_use_doctor(
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON health report"),
) -> None:
    """Run the Computer Use health check matrix (display server, AX framework, SoM compositor, safety guards)."""
    session = get_computer_use_session()
    report = session.health_report()

    if as_json:
        console.print_json(json.dumps(report, indent=2))
        return

    status_badge = "[bold green]HEALTHY[/bold green]" if report["status"] == "healthy" else "[bold yellow]DEGRADED[/bold yellow]"
    console.print(f"\n[bold]Valstorm Agent Computer Use Doctor[/bold] — {status_badge}")
    console.print(f"Platform: [cyan]{report['platform']}[/cyan] | Display: [cyan]{report['display_server']}[/cyan]")
    console.print(f"Permission Mode: [magenta]{report['permission_mode']}[/magenta] | Artifacts: [dim]{report['artifacts_dir']}[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Status", width=10)
    table.add_column("Details")

    for item in report.get("checks", []):
        st = item["status"]
        if st == "pass":
            badge = "[green]PASS[/green]"
        elif st == "info":
            badge = "[blue]INFO[/blue]"
        elif st == "warn":
            badge = "[yellow]WARN[/yellow]"
        else:
            badge = "[red]FAIL[/red]"
        table.add_row(item["check"], badge, item["detail"])

    console.print(table)


@computer_use_app.command(name="status")
def computer_use_status() -> None:
    """Display active Computer Use driver readiness, permission mode, and focused app."""
    session = get_computer_use_session()
    report = session.health_report()
    console.print(
        f"[bold]Computer Use Status:[/bold] {report['status'].upper()} "
        f"(Native Bridge: {'Active' if report['native_bridge_active'] else 'Inactive'}, "
        f"Focused App: {session.focused_app})"
    )


@computer_use_app.command(name="install")
def computer_use_install(
    prompt_permissions: bool = typer.Option(
        True,
        "--prompt-permissions/--no-prompt-permissions",
        help="Trigger native macOS Accessibility & Screen Recording permission dialogs if not yet granted",
    ),
) -> None:
    """Configure local Computer Use artifacts directory, install `~/.valstorm/bin/cua-driver`, and verify OS permissions."""
    bin_dir = Path(os.path.expanduser("~/.valstorm/bin"))
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim_path = bin_dir / "cua-driver"

    shim_content = """#!/usr/bin/env sh
# Valstorm Agent Runtime Built-in CUA Driver Bridge Shim
exec vsagent computer-use doctor "$@"
"""
    shim_path.write_text(shim_content, encoding="utf-8")
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    session = get_computer_use_session()
    report = session.health_report()
    console.print(f"[green]✓[/green] Installed Computer Use driver shim at [bold]{shim_path}[/bold]")
    console.print(f"[green]✓[/green] Verified Set-of-Mark artifacts directory at [bold]{report['artifacts_dir']}[/bold]")

    if prompt_permissions:
        perms = session._macos_check_tcc_permissions(prompt_user=True)
        console.print(
            f"[cyan]•[/cyan] macOS TCC Permissions — Accessibility: [bold]{perms.get('accessibility')}[/bold], "
            f"Screen Recording: [bold]{perms.get('screen_recording')}[/bold]"
        )
