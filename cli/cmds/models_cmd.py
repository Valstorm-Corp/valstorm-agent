"""`vsagent models` — list the models the Valstorm gateway will accept and what they resolve to."""

from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def models_command(
    env: str = typer.Option("prod", "--env", "-e", help="Valstorm environment (prod, dev, local)"),
    token: Optional[str] = typer.Option(None, "--token", help="Valstorm API token override"),
    live: bool = typer.Option(False, "--live", help="Also ask the backend for its live model list (slower)"),
):
    """List valid models on the Valstorm AI gateway (aliases, pinned targets, backend)."""
    from tools.valstorm_client import resolve_valstorm_credentials

    access_token, base_url = resolve_valstorm_credentials(override_token=token, env=env)
    if not access_token or not base_url:
        console.print("[bold red]Not logged in to Valstorm.[/bold red] Run `valstorm login` first.")
        raise typer.Exit(1)

    url = f"{base_url.rstrip('/')}/ai/models"
    auth = access_token if access_token.startswith("Bearer ") else f"Bearer {access_token}"
    try:
        resp = httpx.get(url, headers={"Authorization": auth}, params={"live": str(live).lower()}, timeout=30.0)
    except Exception as e:
        console.print(f"[bold red]Request failed:[/bold red] {e}")
        raise typer.Exit(1)
    if resp.status_code == 404:
        console.print("[yellow]This API does not expose /ai/models yet (deploy the gateway update).[/yellow]")
        raise typer.Exit(1)
    if resp.status_code >= 400:
        console.print(f"[bold red]{resp.status_code}[/bold red] {resp.text[:500]}")
        raise typer.Exit(1)

    data = resp.json()
    console.print(f"[bold]Gemini backend:[/bold] {data.get('gemini_backend')}   "
                  f"[bold]Default thinking level:[/bold] {data.get('default_thinking_level') or 'backend default'}")

    table = Table(title="Model aliases")
    table.add_column("Requested name", style="bold cyan")
    table.add_column("Resolves to")
    table.add_column("Route")
    for row in data.get("aliases", []):
        table.add_row(row.get("name", ""), row.get("target", ""), row.get("route", ""))
    console.print(table)

    live_models = data.get("live_models") or []
    if live_models:
        lt = Table(title=f"Live Gemini models on backend ({len(live_models)})")
        lt.add_column("Model id", style="cyan")
        for m in live_models:
            lt.add_row(m)
        console.print(lt)
    elif live:
        console.print(f"[dim]Live listing unavailable: {data.get('live_error') or 'no data'}[/dim]")
    console.print("[dim]Any concrete `gemini-*`, `claude-*` or `<publisher>/<model>` id enabled for the project is also accepted.[/dim]")
