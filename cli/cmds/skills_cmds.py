"""Skill discovery and inspection commands for Valstorm Agent CLI."""

from typing import Optional
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from tools.skill_tool import _index_all_skills, _resolve_skills_dir, skill_view

skills_app = typer.Typer(help="Inspect and search procedural agent skills", no_args_is_help=False)
console = Console()


@skills_app.callback(invoke_without_command=True)
def default_skills(ctx: typer.Context):
    """List skills if no sub-command is provided."""
    if ctx.invoked_subcommand is None:
        list_skills()


@skills_app.command(name="list")
def list_skills(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum skills to list"),
):
    """List installed procedural skills."""
    skills = _index_all_skills()
    if category:
        skills = [s for s in skills if s["category"].lower() == category.lower()]

    if not skills:
        console.print("[yellow]No skills found.[/yellow]")
        return

    table = Table(title=f"Installed Skills ({len(skills)} total)")
    table.add_column("Category", style="cyan")
    table.add_column("Slug", style="bold white")
    table.add_column("Name", style="green")
    table.add_column("Description", style="dim")

    for s in skills[:limit]:
        desc = (s.get("description") or "").replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(s["category"], s["slug"], s["name"], desc)

    console.print(table)
    if len(skills) > limit:
        console.print(f"[dim]Showing first {limit} of {len(skills)} skills. Use --limit to show more.[/dim]")


@skills_app.command(name="show")
def show_skill(
    slug: str = typer.Argument(..., help="Skill slug (e.g. bash-scripting, test-driven-development)"),
):
    """View full instructions and standard operating procedure for a skill."""
    content = skill_view(name=slug)
    if "not found" in content.lower():
        console.print(f"[bold red]{content}[/bold red]")
        raise typer.Exit(1)
    console.print(Markdown(content))


@skills_app.command(name="search")
def search_skills_cmd(
    query: str = typer.Argument(..., help="Search query keyword"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max search results"),
):
    """Search procedural skills by keyword, name, or description."""
    skills = _index_all_skills()
    q = query.lower()
    matched = []
    for s in skills:
        if (
            q in s["slug"].lower()
            or q in s["name"].lower()
            or q in s["description"].lower()
            or q in s["category"].lower()
            or q in s.get("body_preview", "").lower()
        ):
            matched.append(s)

    if not matched:
        console.print(f"[yellow]No skills matched query '{query}'.[/yellow]")
        return

    table = Table(title=f"Skill Search Results for '{query}' ({len(matched)} matches)")
    table.add_column("Category", style="cyan")
    table.add_column("Slug", style="bold white")
    table.add_column("Name", style="green")
    table.add_column("Description", style="dim")

    for s in matched[:limit]:
        desc = (s.get("description") or "").replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(s["category"], s["slug"], s["name"], desc)

    console.print(table)
