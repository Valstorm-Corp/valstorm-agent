"""Persistent declarative memory commands for Valstorm Agent CLI."""

import json
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from core.memory import MemoryStore

memory_app = typer.Typer(help="Inspect and manage persistent declarative memory", no_args_is_help=False)
console = Console()


@memory_app.callback(invoke_without_command=True)
def default_memory(ctx: typer.Context):
    """List memory facts if no sub-command is provided."""
    if ctx.invoked_subcommand is None:
        list_memory()


@memory_app.command(name="list")
def list_memory(
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Filter target ('user' or 'memory')"),
):
    """List persistent declarative facts stored in agent memory."""
    store = MemoryStore()
    facts = store.get_facts(target=target)

    if not facts:
        console.print("[yellow]No declarative memory facts stored yet.[/yellow]")
        return

    table = Table(title="Declarative Long-Term Memory Facts")
    table.add_column("Category", style="cyan")
    table.add_column("Fact / Note", style="white")

    if isinstance(facts, dict):
        for cat, items in facts.items():
            if target and cat != target:
                continue
            if isinstance(items, list):
                for item in items:
                    table.add_row(cat, str(item))
            elif isinstance(items, dict):
                for k, v in items.items():
                    table.add_row(cat, f"{k}: {v}")
    elif isinstance(facts, list):
        for item in facts:
            table.add_row(target or "memory", str(item))

    console.print(table)


@memory_app.command(name="add")
def add_memory(
    fact: str = typer.Argument(..., help="The fact or rule to store"),
    target: str = typer.Option("memory", "--target", "-t", help="Target partition ('memory' or 'user')"),
):
    """Add a new persistent fact to agent memory."""
    store = MemoryStore()
    if hasattr(store, "add_fact"):
        store.add_fact(fact=fact, target=target)
    elif hasattr(store, "save_fact"):
        store.save_fact(fact=fact, target=target)
    console.print(f"[bold green]✓[/bold green] Saved fact to [{target}]: '{fact}'")


@memory_app.command(name="remove")
def remove_memory(
    pattern: str = typer.Argument(..., help="Substring pattern of fact to remove"),
    target: str = typer.Option("memory", "--target", "-t", help="Target partition ('memory' or 'user')"),
):
    """Remove a fact matching substring from agent memory."""
    store = MemoryStore()
    if hasattr(store, "remove_fact"):
        success = store.remove_fact(old_text=pattern, target=target)
        if success:
            console.print(f"[bold green]✓[/bold green] Removed fact matching '{pattern}'.")
        else:
            console.print(f"[yellow]No fact matching '{pattern}' found.[/yellow]")
    else:
        console.print("[yellow]Memory removal not supported in current store.[/yellow]")


@memory_app.command(name="clear")
def clear_memory(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirm clear without prompting"),
):
    """Clear all declarative memory facts."""
    if not confirm and not typer.confirm("Are you sure you want to clear all agent memory?"):
        raise typer.Exit()
    store = MemoryStore()
    if hasattr(store, "clear"):
        store.clear()
        console.print("[bold green]✓[/bold green] Memory cleared.")
    else:
        console.print("[yellow]Clear not supported in current store.[/yellow]")
