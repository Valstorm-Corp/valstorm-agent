"""API Key management commands for Valstorm Agent CLI."""

from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from core.keystore import KeyStore

keys_app = typer.Typer(help="Manage and inspect provider API keys", no_args_is_help=False)
console = Console()


@keys_app.callback(invoke_without_command=True)
def default_keys(ctx: typer.Context):
    """Show API key status if no sub-command is provided."""
    if ctx.invoked_subcommand is None:
        status_keys()


@keys_app.command(name="list")
@keys_app.command(name="status")
def status_keys():
    """Check configuration status of all LLM and platform API keys."""
    keystore = KeyStore()
    providers = [
        "gemini",
        "digitalocean",
        "deepseek",
        "kimi",
        "openai",
        "anthropic",
        "valstorm",
    ]

    table = Table(title="Configured API Keys")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Status", style="white")
    table.add_column("Key Preview", style="dim")

    for prov in providers:
        key = keystore.get_api_key(prov)
        if not key and prov == "valstorm":
            try:
                from tools.valstorm_client import resolve_valstorm_credentials
                key, _ = resolve_valstorm_credentials()
            except Exception:
                pass
        if key:
            status_text = "[bold green]Configured[/bold green]"
            preview = f"{key[:4]}...{key[-4:]}" if len(key) >= 8 else "****"
        else:
            status_text = "[bold red]Missing[/bold red]"
            preview = "-"
        table.add_row(prov.capitalize(), status_text, preview)

    console.print(table)


@keys_app.command(name="set")
def set_key(
    provider: str = typer.Argument(..., help="Provider name (gemini, openai, anthropic, valstorm)"),
    key: str = typer.Argument(..., help="API key string"),
):
    """Save an API key to persistent local configuration (~/.config/valstorm/keys.json)."""
    keystore = KeyStore()
    prov = provider.lower()
    saved = keystore.save_api_key(prov, key, persist_to="config")
    console.print(f"[bold green]✓[/bold green] API key for '{prov}' successfully saved to [cyan]{saved}[/cyan].")
