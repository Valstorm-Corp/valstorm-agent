"""System status and CLI self-updater commands for Valstorm Agent CLI."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil
import subprocess
import sys
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.context import WorkspaceContextManager, list_available_profiles
from core.keystore import KeyStore
from core.memory import MemoryStore
from core.storage import SessionStore
from tools.skill_tool import _index_all_skills

console = Console()


def get_version_string() -> str:
    """Returns the installed or local version of valstorm-agent."""
    try:
        return version("valstorm-agent")
    except PackageNotFoundError:
        try:
            current = Path(__file__).resolve()
            for p in [current, *current.parents]:
                pyproject = p / "pyproject.toml"
                if pyproject.exists():
                    import re
                    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', pyproject.read_text())
                    if match:
                        return match.group(1)
        except Exception:
            pass
        return "2.1.1"


def version_callback(value: bool):
    """Callback for --version flag."""
    if value:
        ver = get_version_string()
        console.print(f"vsagent v{ver}")
        raise typer.Exit()


def version_command():
    """Display vsagent version."""
    ver = get_version_string()
    console.print(f"vsagent v{ver}")


def status_command():
    """Display system status, available profiles, skills count, sessions, and keys."""
    keystore = KeyStore()
    session_store = SessionStore()
    memory_store = MemoryStore()
    context_mgr = WorkspaceContextManager()

    ver = get_version_string()

    console.print(f"\n[bold cyan]Valstorm Agent Runtime (vsagent) v{ver}[/bold cyan]")
    console.print(f"[dim]Working Directory: {context_mgr.workdir}[/dim]\n")

    # 1. API Keys Table
    table_keys = Table(title="Provider API Keys")
    table_keys.add_column("Provider", style="cyan")
    table_keys.add_column("Status", style="white")
    table_keys.add_column("Key Preview", style="dim")

    for prov in ["gemini", "openai", "anthropic", "valstorm"]:
        k = keystore.get_api_key(prov)
        if not k and prov == "valstorm":
            try:
                from tools.valstorm_client import resolve_valstorm_credentials
                k, _ = resolve_valstorm_credentials()
            except Exception:
                pass
        if k:
            status_text = "[bold green]Configured[/bold green]"
            preview = f"{k[:4]}...{k[-4:]}" if len(k) >= 8 else "****"
        else:
            status_text = "[bold red]Missing[/bold red]"
            preview = "-"
        table_keys.add_row(prov.capitalize(), status_text, preview)

    console.print(table_keys)

    # 2. Workspace & Data Summary
    profiles = list_available_profiles()
    skills = _index_all_skills()
    sessions = session_store.list_sessions(limit=5)
    facts = memory_store.get_facts()
    fact_count = sum(len(v) if isinstance(v, list) else 1 for v in facts.values()) if isinstance(facts, dict) else len(facts) if isinstance(facts, list) else 0

    summary_text = (
        f"• Available Agent Profiles: [bold cyan]{len(profiles)}[/bold cyan] ({', '.join(p.get('api_name', p.get('name')) for p in profiles[:6])}...)\n"
        f"• Installed Procedural Skills: [bold cyan]{len(skills)}[/bold cyan] skills\n"
        f"• Stored Conversation Sessions: [bold cyan]{len(sessions)}[/bold cyan] recent sessions\n"
        f"• Declarative Memory Facts: [bold cyan]{fact_count}[/bold cyan] facts"
    )
    console.print(Panel(summary_text, title="Workspace & Agent Subsystems", border_style="cyan"))


def update_command():
    """Update the vsagent CLI to the latest version."""
    try:
        ver = version("valstorm-agent")
    except PackageNotFoundError:
        ver = "local-dev"

    console.print(f"Current version: [cyan]{ver}[/cyan]")
    repo_url = "git+https://github.com/Valstorm-Corp/monorepo.git#subdirectory=apps/agent-runtime"

    try:
        with console.status("[bold cyan]Updating vsagent CLI...[/bold cyan]"):
            # Check if running inside local monorepo source tree
            is_installed = "site-packages" in __file__
            if not is_installed:
                # Find monorepo root
                current = Path(__file__).resolve()
                for p in [current, *current.parents]:
                    if (p / "apps" / "agent-runtime" / "pyproject.toml").exists():
                        if shutil.which("uv"):
                            subprocess.run(["uv", "sync", "--project", str(p / "apps" / "agent-runtime")], check=True, capture_output=True)
                            subprocess.run(["uv", "tool", "install", "--force", str(p / "apps" / "agent-runtime")], check=True, capture_output=True)
                            console.print("[bold green]✓[/bold green] Local vsagent updated and reinstalled globally.")
                            return

            # Check if uv tool
            if shutil.which("uv") and "uv/tools" in sys.executable:
                subprocess.run(["uv", "tool", "upgrade", "valstorm-agent"], check=True, capture_output=True)
                console.print("[bold green]✓[/bold green] vsagent uv tool updated successfully.")
                return

            if shutil.which("uv"):
                subprocess.run(["uv", "pip", "install", "--upgrade", repo_url], check=True, capture_output=True)
            else:
                subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", repo_url], check=True, capture_output=True)

        console.print("[bold green]✓[/bold green] vsagent updated successfully.")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Update failed:[/bold red] {e}")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        if err:
            console.print(err[-3000:])
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Update failed:[/bold red] {e}")
        raise typer.Exit(1)
