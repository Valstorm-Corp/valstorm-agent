"""CLI Commands for Inspecting, Pushing, and Pulling Agent Profiles."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.context import (
    BUILTIN_PROFILES,
    _resolve_profiles_dir,
    list_available_profiles,
    load_profile,
)
from tools.valstorm_client import ValstormApiClient


def _get_valstorm_config_path() -> Path:
    # Typically ~/.valstorm/config.json stores the current active environment set by 'valstorm auth login'
    return Path(os.environ.get("VALSTORM_CONFIG_HOME") or Path.home() / ".valstorm") / "config.json"


def _get_default_env_slug() -> str:
    # 1. Check environment variable override
    env_override = os.environ.get("VALSTORM_ENV")
    if env_override:
        return env_override.lower()
    
    # 2. Check global config file
    try:
        with open(_get_valstorm_config_path(), "r") as f:
            config = json.load(f)
            # The environment variable is "env" in the config file, as confirmed in core/context.py
            return config.get("env", "local").lower()
    except Exception:
        # 3. Fallback default
        return "local"


console = Console()
profiles_app = typer.Typer(help="Manage, sync, and inspect agent profiles", no_args_is_help=False)


def _normalize_provider(val: Optional[str]) -> str:
    if not val:
        return "Gemini"
    val = val.strip()
    if val.lower() == "gemini":
        return "Gemini"
    elif val.lower() == "claude":
        return "Claude"
    elif val.lower() in ("chatgpt", "openai", "gpt"):
        return "ChatGPT"
    return val.capitalize()


@profiles_app.callback(invoke_without_command=True)
def default_profiles(ctx: typer.Context):
    """List agent profiles if no sub-command is provided."""
    if ctx.invoked_subcommand is None:
        list_profiles()


@profiles_app.command(name="list")
def list_profiles():
    """List all available local agent profiles."""
    profs = list_available_profiles()
    table = Table(title="Available Valstorm Agent Profiles (Local)")
    table.add_column("Slug", style="bold cyan")
    table.add_column("Display Name", style="white")
    table.add_column("Tier", style="green")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Tools", style="yellow")
    table.add_column("Skills", style="blue")

    for p in profs:
        slug = p.get("api_name", p.get("name", "unknown"))
        name = p.get("name") or p.get("display_name") or slug.title()
        tier = p.get("model_tier", "tier_2")
        provider = p.get("provider", "Gemini")
        model = p.get("model", "gemini-flash-latest")
        tools_cnt = str(len(p.get("allowed_tools", []))) if p.get("allowed_tools") else "all"
        skills_cnt = str(len(p.get("ai_skills", []) or p.get("attached_skill_slugs", [])))
        table.add_row(slug, name, tier, provider, model, tools_cnt, skills_cnt)

    console.print(table)


@profiles_app.command(name="show")
def show_profile(
    slug: str = typer.Argument(..., help="Profile slug (e.g. developer, researcher, orchestrator)"),
):
    """Display full details, prompt, tools, and skills for a specific profile."""
    prof = load_profile(slug)
    if not prof:
        console.print(f"[bold red]Error:[/bold red] Profile '{slug}' not found.")
        raise typer.Exit(1)

    name = prof.get("name", slug)
    desc = prof.get("description", "No description")
    tier = prof.get("model_tier", "tier_2")
    model = prof.get("model", "gemini-flash-latest")
    provider = prof.get("provider", "Gemini")
    tools = prof.get("allowed_tools", ["<all>"])
    skills = prof.get("ai_skills", []) or prof.get("attached_skill_slugs", [])
    tags = prof.get("tag", [])
    paths = prof.get("scoped_paths", [])
    system_prompt = prof.get("system_prompt", "Default role prompt")

    console.print(f"\n[bold cyan]Profile: {name}[/bold cyan] ([dim]{slug}[/dim])")
    console.print(f"[bold]Description:[/bold] {desc}")
    console.print(f"[bold]Tier:[/bold] {tier} | [bold]Provider:[/bold] {provider} | [bold]Model:[/bold] {model}")
    if tags:
        console.print(f"[bold]Knowledge Graph Tags ({len(tags)}):[/bold] {', '.join(tags)}")
    if paths:
        console.print(f"[bold]Scoped Paths ({len(paths)}):[/bold] {', '.join(paths)}")
    console.print(f"[bold]Allowed Tools ({len(tools)}):[/bold] {', '.join(tools)}")
    console.print(f"[bold]Attached Skills ({len(skills)}):[/bold] {', '.join(skills) if skills else 'None'}")
    console.print("\n[bold]System Prompt:[/bold]")
    console.print(Panel(system_prompt, title=f"System Prompt ({slug})", border_style="cyan"))


async def _fetch_remote_agents(client: ValstormApiClient) -> Dict[str, Dict[str, Any]]:
    """Helper to query all remote ai_agent records indexed by api_name and ID."""
    res = await client.sql_query(
        "SELECT id, name, api_name, model_tier, model, provider, allowed_tools, description, system_prompt, ai_skills, tag, scoped_paths FROM ai_agent",
        bypass_cache=True,
    )
    records = res if isinstance(res, list) else res.get("records", [])
    remote_map = {}
    for r in records:
        api_name = r.get("api_name")
        if api_name:
            remote_map[api_name.lower().strip()] = r
        if r.get("id"):
            remote_map[r["id"]] = r
    return remote_map


@profiles_app.command(name="diff")
def diff_profiles(
    env: str = typer.Option(_get_default_env_slug, "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    profile_slug: Optional[str] = typer.Option(None, "--profile", "-p", help="Filter diff to a specific profile slug"),
):
    """Compare local profile configurations against cloud database records."""
    async def _run():
        client = ValstormApiClient(env=env)
        try:
            remote_map = await _fetch_remote_agents(client)
        except Exception as e:
            console.print(f"[bold red]Error querying Valstorm ({env}):[/bold red] {e}")
            raise typer.Exit(1)

        local_profiles = list_available_profiles()
        if profile_slug:
            local_profiles = [p for p in local_profiles if p.get("api_name") == profile_slug or p.get("name") == profile_slug]

        table = Table(title=f"Agent Profile Alignment Diff (Target: [{env.upper()}])")
        table.add_column("Profile Slug", style="bold cyan")
        table.add_column("Local Tier", style="green")
        table.add_column("Remote Tier", style="green")
        table.add_column("Local Model", style="magenta")
        table.add_column("Remote Model", style="magenta")
        table.add_column("Status", style="bold")

        diff_count = 0
        for lp in local_profiles:
            slug = (lp.get("api_name") or lp.get("name", "")).lower().strip()
            loc_tier = lp.get("model_tier", "tier_2")
            loc_model = lp.get("model", "gemini-flash-latest")
            loc_prov = _normalize_provider(lp.get("provider"))

            remote = remote_map.get(slug)
            if not remote:
                table.add_row(slug, loc_tier, "-", loc_model, "-", "[yellow]Missing on Remote[/yellow]")
                diff_count += 1
                continue

            rem_tier = remote.get("model_tier", "-")
            rem_model = remote.get("model", "-")
            rem_prov = _normalize_provider(remote.get("provider"))

            if loc_model != rem_model or loc_prov != rem_prov or loc_tier != rem_tier:
                table.add_row(slug, loc_tier, str(rem_tier), loc_model, str(rem_model), "[red]Mismatch[/red]")
                diff_count += 1
            else:
                table.add_row(slug, loc_tier, str(rem_tier), loc_model, str(rem_model), "[green]Aligned[/green]")

        console.print(table)
        if diff_count > 0:
            console.print(f"\n[bold yellow]Found {diff_count} profile alignment difference(s).[/bold yellow]")
            console.print(f"Run [cyan]vsagent profiles pull --env {env} --execute[/cyan] or [cyan]vsagent profiles push --env {env} --execute[/cyan] to align.")
        else:
            console.print(f"\n[bold green]All local profiles are in sync with [{env.upper()}].[/bold green]")

    asyncio.run(_run())


@profiles_app.command(name="push")
def push_profiles(
    env: str = typer.Option(_get_default_env_slug, "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    profile_slug: Optional[str] = typer.Option(None, "--profile", "-p", help="Push a specific profile slug only"),
    execute: bool = typer.Option(False, "--execute", help="Execute database mutations (defaults to dry-run preview)"),
):
    """Push local profile models, tiers, tools, and configs to the Valstorm database."""
    async def _run():
        console.print(f"📦 [bold]Pushing Local Profiles to Valstorm [{env.upper()}][/bold] (execute={execute})\n")
        client = ValstormApiClient(env=env)
        try:
            remote_map = await _fetch_remote_agents(client)
        except Exception as e:
            console.print(f"[bold red]Error querying Valstorm ({env}):[/bold red] {e}")
            raise typer.Exit(1)

        local_profiles = list_available_profiles()
        if profile_slug:
            local_profiles = [p for p in local_profiles if p.get("api_name") == profile_slug or p.get("name") == profile_slug]

        updates = []
        creates = []

        for lp in local_profiles:
            slug = (lp.get("api_name") or lp.get("name", "")).lower().strip()
            loc_tier = lp.get("model_tier", "tier_2")
            loc_model = lp.get("model", "gemini-flash-latest")
            loc_prov = _normalize_provider(lp.get("provider"))
            loc_desc = lp.get("description", "")
            loc_tools = lp.get("allowed_tools", [])
            loc_skills = lp.get("ai_skills", [])
            loc_prompt = lp.get("system_prompt", "")
            loc_name = lp.get("name") or slug.title()
            loc_tag = lp.get("tag", [])
            loc_scoped_paths = lp.get("scoped_paths", [])

            remote = remote_map.get(slug)
            if remote:
                rem_id = remote.get("id")
                rem_tier = remote.get("model_tier")
                rem_model = remote.get("model")
                rem_prov = _normalize_provider(remote.get("provider"))
                rem_tag = remote.get("tag", []) or []
                rem_scoped_paths = remote.get("scoped_paths", []) or []

                if rem_model != loc_model or rem_prov != loc_prov or rem_tier != loc_tier or rem_tag != loc_tag or rem_scoped_paths != loc_scoped_paths:
                    console.print(f"  • [yellow]Update[/yellow] [bold]{loc_name}[/bold] ({slug}): {rem_tier}/{rem_model} -> [bold green]{loc_tier}/{loc_model} ({loc_prov})[/bold green]")
                    updates.append({
                        "id": rem_id,
                        "name": loc_name,
                        "api_name": slug,
                        "model_tier": loc_tier,
                        "model": loc_model,
                        "provider": loc_prov,
                        "description": loc_desc,
                        "allowed_tools": loc_tools,
                        "ai_skills": loc_skills,
                        "system_prompt": loc_prompt,
                        "tag": loc_tag,
                        "scoped_paths": loc_scoped_paths,
                        "is_active": True,
                    })
            else:
                console.print(f"  • [cyan]Create[/cyan] [bold]{loc_name}[/bold] ({slug}) with tier [bold green]{loc_tier}[/bold green] & model [bold green]{loc_model}[/bold green]")
                creates.append({
                    "name": loc_name,
                    "api_name": slug,
                    "model_tier": loc_tier,
                    "model": loc_model,
                    "provider": loc_prov,
                    "description": loc_desc,
                    "allowed_tools": loc_tools,
                    "ai_skills": loc_skills,
                    "system_prompt": loc_prompt,
                    "tag": loc_tag,
                    "scoped_paths": loc_scoped_paths,
                    "is_active": True,
                })

        if not updates and not creates:
            console.print("[bold green]All profiles are already in sync! No changes needed.[/bold green]")
            return

        if not execute:
            console.print(f"\n[bold yellow]Dry-run complete. {len(updates)} update(s) and {len(creates)} create(s) pending.[/bold yellow]")
            console.print(f"Run with [cyan]--execute[/cyan] to apply changes to [{env.upper()}].")
            return

        if updates:
            console.print(f"\n🚀 Sending batch updates for {len(updates)} profile(s)...")
            res_up = await client.records_update("ai_agent", updates)
            console.print(f"  ✅ Updated {len(updates)} profile(s) on Valstorm.")

        if creates:
            console.print(f"\n🚀 Sending batch creates for {len(creates)} profile(s)...")
            res_cr = await client.records_create("ai_agent", creates)
            console.print(f"  ✅ Created {len(creates)} profile(s) on Valstorm.")

        console.print(f"\n[bold green]Successfully synchronized profiles to [{env.upper()}].[/bold green]")

    asyncio.run(_run())


@profiles_app.command(name="pull")
def pull_profiles(
    env: str = typer.Option(_get_default_env_slug, "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    profile_slug: Optional[str] = typer.Option(None, "--profile", "-p", help="Pull a specific profile slug only"),
    execute: bool = typer.Option(False, "--execute", help="Write changes to disk (defaults to dry-run preview)"),
):
    """Pull remote cloud agent profiles down into local profile JSON files."""
    async def _run():
        console.print(f"📥 [bold]Pulling Profiles from Valstorm [{env.upper()}][/bold] (execute={execute})\n")
        client = ValstormApiClient(env=env)
        try:
            remote_map = await _fetch_remote_agents(client)
        except Exception as e:
            console.print(f"[bold red]Error querying Valstorm ({env}):[/bold red] {e}")
            raise typer.Exit(1)

        prof_dir = _resolve_profiles_dir()
        prof_dir.mkdir(parents=True, exist_ok=True)

        pulled_count = 0
        for key, record in remote_map.items():
            slug = record.get("api_name")
            if not slug or key != slug.lower().strip():
                continue

            if profile_slug and slug != profile_slug:
                continue

            dest_file = prof_dir / f"{slug}.json"
            console.print(f"  • [cyan]Pulling[/cyan] [bold]{record.get('name', slug)}[/bold] ({slug}) -> tier: {record.get('model_tier', 'tier_2')} | model: {record.get('model')}")

            if execute:
                existing_data = {}
                if dest_file.is_file():
                    try:
                        existing_data = json.loads(dest_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                existing_data.update({
                    "id": record.get("id") or existing_data.get("id"),
                    "name": record.get("name") or existing_data.get("name"),
                    "api_name": slug,
                    "description": record.get("description") or existing_data.get("description"),
                    "model_tier": record.get("model_tier") or existing_data.get("model_tier", "tier_2"),
                    "model": record.get("model") or existing_data.get("model"),
                    "provider": _normalize_provider(record.get("provider")),
                    "allowed_tools": record.get("allowed_tools") or existing_data.get("allowed_tools"),
                    "ai_skills": record.get("ai_skills") or existing_data.get("ai_skills", []),
                    "system_prompt": record.get("system_prompt") or existing_data.get("system_prompt"),
                    "tag": record.get("tag") or existing_data.get("tag", []),
                    "scoped_paths": record.get("scoped_paths") or existing_data.get("scoped_paths", []),
                    "is_active": record.get("is_active", True),
                })
                dest_file.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")
                pulled_count += 1

        if not execute:
            console.print(f"\n[bold yellow]Dry-run complete. Run with `--execute` to write profile files to {prof_dir}.[/bold yellow]")
        else:
            console.print(f"\n[bold green]Successfully pulled and updated {pulled_count} local profile(s) in {prof_dir}.[/bold green]")

    asyncio.run(_run())


@profiles_app.command(name="sync")
def sync_profiles(
    env: str = typer.Option(_get_default_env_slug, "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    profile_slug: Optional[str] = typer.Option(None, "--profile", "-p", help="Sync a specific profile slug only"),
    execute: bool = typer.Option(False, "--execute", help="Apply synchronization changes (defaults to dry-run)"),
):
    """Synchronize local profiles with Valstorm cloud database."""
    push_profiles(env=env, profile_slug=profile_slug, execute=execute)
