"""Session management commands for Valstorm Agent CLI."""

from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from core.storage import SessionStore

sessions_app = typer.Typer(help="Manage and search past agent sessions", no_args_is_help=False)
console = Console()


@sessions_app.callback(invoke_without_command=True)
def default_sessions(ctx: typer.Context):
    """List recent sessions if no sub-command is provided."""
    if ctx.invoked_subcommand is None:
        list_sessions()


@sessions_app.command(name="list")
def list_sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to list"),
):
    """List recent agent conversation sessions."""
    store = SessionStore()
    limit_int = 20 if not isinstance(limit, int) else limit
    sessions = store.list_sessions(limit=limit_int)

    if not sessions:
        console.print("[yellow]No past sessions stored.[/yellow]")
        return

    table = Table(title=f"Recent Agent Sessions ({len(sessions)} shown)")
    table.add_column("Session ID", style="bold cyan")
    table.add_column("Title / Prompt", style="white")
    table.add_column("Messages", style="magenta")
    table.add_column("Last Updated", style="dim")

    for s in sessions:
        table.add_row(
            s["session_id"][:12],
            s.get("title", "Untitled")[:40],
            str(s.get("message_count", 0)),
            s.get("updated_at", "")[:16],
        )

    console.print(table)


@sessions_app.command(name="show")
def show_session(
    session_id: str = typer.Argument(..., help="Session ID (or prefix)"),
):
    """Display messages and turns from a past session."""
    store = SessionStore()
    all_sess = store.list_sessions(limit=100)
    matched = [s["session_id"] for s in all_sess if s["session_id"].startswith(session_id)]
    target_id = matched[0] if matched else session_id

    session = store.load_session(target_id)
    if not session:
        console.print(f"[bold red]Error:[/bold red] Session '{session_id}' not found.")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Session:[/bold cyan] {session.session_id}")
    console.print(f"[bold]Model:[/bold] {session.active_model} | [bold]Provider:[/bold] {session.active_provider}")
    console.print(f"[bold]Total Tokens:[/bold] {session.total_tokens:,} (Prompt: {session.total_prompt_tokens:,} | Completion: {session.total_completion_tokens:,})")
    console.print(f"[dim]To resume:[/dim] vsagent chat --session {session.session_id}")
    console.print(f"[bold]Messages ({len(session.messages)}):[/bold]\n")

    for idx, msg in enumerate(session.messages, 1):
        role_style = "green" if msg.role == "assistant" else "cyan" if msg.role == "user" else "yellow"
        content_preview = (msg.content or msg.body or "")[:120]
        if msg.tool_calls:
            content_preview += f" [Tools: {', '.join(tc.name for tc in msg.tool_calls)}]"
        console.print(f"{idx:2d}. [{role_style}][bold]{msg.role.upper():9}[/bold][/{role_style}] : {content_preview}")


@sessions_app.command(name="search")
def search_sessions_cmd(
    query: str = typer.Argument(..., help="Search query string"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results to return"),
):
    """Search past conversation turns and tool results via SQLite FTS5."""
    store = SessionStore()
    results = store.search_sessions(query=query, limit=limit)

    if not results:
        console.print(f"[yellow]No sessions matched query '{query}'.[/yellow]")
        return

    console.print(f"\n[bold]FTS5 Search Results for '{query}':[/bold]")
    for r in results:
        console.print(f"  • [bold cyan]{r['title']}[/bold cyan] ({r['session_id'][:8]} · [dim]{r['updated_at'][:10]}[/dim]):")
        console.print(f"    [dim]{r['snippet']}[/dim]\n")
