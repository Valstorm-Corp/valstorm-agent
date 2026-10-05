"""CLI Commands for Managing, Validating, and Bi-directionally Syncing Markdown Catalog.

Provides:
- vsagent catalog list [--visibility all|public|internal]
- vsagent catalog validate / vsagent catalog lint
- vsagent catalog build-profiles [--output-dir ~/.valstorm/profiles]
- vsagent catalog push [--visibility all|public|internal] [--dry-run] [--env local|dev|prod] [--valstorm-internal-org org_dSnPMRjS1ZkkdYQ2]
- vsagent catalog pull [--visibility all|public|internal] [--env local|dev|prod]
- vsagent catalog cleanup / vsagent catalog reorganize [--cloud-sync] [--dry-run]
"""

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tools.valstorm_client import ValstormApiClient

try:
    import yaml
except ImportError:
    yaml = None

console = Console()
catalog_app = typer.Typer(
    help="Manage, validate, and bi-directionally sync Markdown skills and agent catalog with Valstorm Cloud",
    no_args_is_help=False,
)

VALSTORM_INTERNAL_ORG_ID = "org_dSnPMRjS1ZkkdYQ2"
VALID_VISIBILITIES = {"public", "internal", "private"}
VALID_MODEL_TIERS = {"tier_1", "tier_2", "tier_3"}


def _find_monorepo_root() -> Path:
    """Discovers monorepo root directory."""
    curr = Path.cwd().resolve()
    for directory in [curr, *curr.parents]:
        if (directory / "skills").is_dir() or (directory / "agents").is_dir() or (directory / "valstorm.json").is_file():
            return directory
    return curr


def _dump_frontmatter(metadata: Dict[str, Any]) -> str:
    """Serializes frontmatter dict into YAML format."""
    if yaml is not None:
        yaml_str = yaml.safe_dump(metadata, sort_keys=False, default_flow_style=False, allow_unicode=True)
        return f"---\n{yaml_str}---\n"

    lines = ["---"]
    for k, v in metadata.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif isinstance(v, str):
            if "\n" in v:
                lines.append(f"{k}: |")
                for subline in v.splitlines():
                    lines.append(f"  {subline}")
            elif ":" in v or "#" in v or '"' in v or "'" in v or v == "":
                escaped = v.replace('"', '\\"')
                lines.append(f'{k}: "{escaped}"')
            else:
                lines.append(f"{k}: {v}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for sub_k, sub_v in v.items():
                lines.append(f"  {sub_k}: {sub_v}")
        elif v is None:
            lines.append(f"{k}: null")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _parse_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Extracts YAML frontmatter dict and markdown body text."""
    if not raw_text.startswith("---"):
        return {}, raw_text.strip()

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text.strip()

    fm_raw = parts[1].strip()
    body = parts[2].strip()

    if yaml is not None:
        try:
            meta = yaml.safe_load(fm_raw) or {}
            if isinstance(meta, dict):
                return meta, body
        except Exception:
            pass

    meta: Dict[str, Any] = {}
    current_list_key = None
    for line in fm_raw.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue
        if line_str.startswith("- ") and current_list_key:
            item = line_str[2:].strip().strip("'\"")
            meta[current_list_key].append(item)
            continue
        if ":" in line_str:
            k, v = line_str.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if not v:
                meta[k] = []
                current_list_key = k
            else:
                current_list_key = None
                if v.lower() == "true":
                    meta[k] = True
                elif v.lower() == "false":
                    meta[k] = False
                elif v.isdigit():
                    meta[k] = int(v)
                else:
                    meta[k] = v

    return meta, body


def load_local_skills_catalog(
    monorepo_root: Optional[Path] = None,
    visibility_filter: str = "all",
) -> Dict[str, Dict[str, Any]]:
    """Discovers and parses all Markdown skills in skills/public/ and skills/internal/."""
    root = monorepo_root or _find_monorepo_root()
    skills_root = root / "skills"
    catalog: Dict[str, Dict[str, Any]] = {}

    if not skills_root.is_dir():
        return catalog

    dirs_to_scan = []
    vis = visibility_filter.lower().strip()

    if vis in ("all", "public"):
        dirs_to_scan.append((skills_root / "public", "public"))
    if vis in ("all", "internal", "private"):
        dirs_to_scan.append((skills_root / "internal", "internal"))

    for base_dir, default_vis in dirs_to_scan:
        if not base_dir.is_dir():
            continue
        for md_file in sorted(base_dir.glob("*/*.md")):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                meta, body = _parse_frontmatter(content)
                slug = meta.get("slug") or md_file.stem
                category = meta.get("category") or md_file.parent.name
                file_vis = meta.get("visibility") or default_vis

                if vis != "all" and file_vis != vis and not (vis == "internal" and file_vis in ("internal", "private")):
                    continue

                catalog[slug] = {
                    "slug": slug,
                    "name": meta.get("name") or slug.replace("-", " ").title(),
                    "api_name": meta.get("api_name") or slug.replace("-", "_"),
                    "category": category,
                    "description": meta.get("description", ""),
                    "visibility": file_vis,
                    "is_active": meta.get("is_active", True),
                    "version": meta.get("version", "1.0.0"),
                    "tags": meta.get("tags", []),
                    "body": body,
                    "file_path": md_file,
                    "meta": meta,
                }
            except Exception as e:
                console.print(f"[bold red]Error parsing skill at {md_file}:[/bold red] {e}")

    return catalog


def load_local_agents_catalog(
    monorepo_root: Optional[Path] = None,
    visibility_filter: str = "all",
) -> Dict[str, Dict[str, Any]]:
    """Discovers and parses all Markdown agent profiles in agents/public/ and agents/internal/."""
    root = monorepo_root or _find_monorepo_root()
    agents_root = root / "agents"
    catalog: Dict[str, Dict[str, Any]] = {}

    if not agents_root.is_dir():
        return catalog

    dirs_to_scan = []
    vis = visibility_filter.lower().strip()

    if vis in ("all", "public"):
        dirs_to_scan.append((agents_root / "public", "public"))
    if vis in ("all", "internal", "private"):
        dirs_to_scan.append((agents_root / "internal", "internal"))

    for base_dir, default_vis in dirs_to_scan:
        if not base_dir.is_dir():
            continue
        for md_file in sorted(base_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                meta, body = _parse_frontmatter(content)
                slug = meta.get("slug") or md_file.stem
                file_vis = meta.get("visibility") or default_vis

                if vis != "all" and file_vis != vis and not (vis == "internal" and file_vis in ("internal", "private")):
                    continue

                catalog[slug] = {
                    "slug": slug,
                    "name": meta.get("name") or slug.replace("-", " ").title(),
                    "api_name": meta.get("api_name") or slug.replace("-", "_"),
                    "description": meta.get("description", ""),
                    "visibility": file_vis,
                    "model_tier": meta.get("model_tier", "tier_2"),
                    "model": meta.get("model", "gemini-flash-latest"),
                    "provider": meta.get("provider", "gemini"),
                    "is_active": meta.get("is_active", True),
                    "allowed_tools": meta.get("allowed_tools", []),
                    "skills": meta.get("skills", []),
                    "tag": meta.get("tag", []),
                    "scoped_paths": meta.get("scoped_paths", []),
                    "system_prompt": body,
                    "file_path": md_file,
                    "meta": meta,
                }
            except Exception as e:
                console.print(f"[bold red]Error parsing agent profile at {md_file}:[/bold red] {e}")

    return catalog


def validate_catalog_integrity(
    monorepo_root: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Lints and validates Markdown skills and agent catalog files."""
    root = monorepo_root or _find_monorepo_root()
    skills = load_local_skills_catalog(root, visibility_filter="all")
    agents = load_local_agents_catalog(root, visibility_filter="all")

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Validate Skills
    for slug, s in skills.items():
        fpath = s["file_path"]
        meta = s["meta"]

        for req in ["name", "slug", "category", "description", "visibility"]:
            if not meta.get(req):
                errors.append(f"[Skill: {slug}] Missing required frontmatter field '{req}' in {fpath}")

        vis = s.get("visibility")
        if vis not in VALID_VISIBILITIES:
            errors.append(f"[Skill: {slug}] Invalid visibility '{vis}' (must be one of {VALID_VISIBILITIES})")

        # Directory segregation check
        if "/skills/public/" in str(fpath) and vis != "public":
            errors.append(f"[Skill: {slug}] Located in skills/public/ but has visibility='{vis}'")
        elif "/skills/internal/" in str(fpath) and vis not in ("internal", "private"):
            errors.append(f"[Skill: {slug}] Located in skills/internal/ but has visibility='{vis}'")

        if not s.get("body", "").strip():
            warnings.append(f"[Skill: {slug}] Body content is empty in {fpath}")

    # 2. Validate Agents
    for slug, a in agents.items():
        fpath = a["file_path"]
        meta = a["meta"]

        for req in ["name", "slug", "visibility", "model_tier", "allowed_tools"]:
            if req not in meta or meta.get(req) is None:
                errors.append(f"[Agent: {slug}] Missing required frontmatter field '{req}' in {fpath}")

        vis = a.get("visibility")
        if vis not in VALID_VISIBILITIES:
            errors.append(f"[Agent: {slug}] Invalid visibility '{vis}' (must be one of {VALID_VISIBILITIES})")

        tier = a.get("model_tier")
        if tier not in VALID_MODEL_TIERS:
            errors.append(f"[Agent: {slug}] Invalid model_tier '{tier}' (must be one of {VALID_MODEL_TIERS})")

        # Directory segregation check
        if "/agents/public/" in str(fpath) and vis != "public":
            errors.append(f"[Agent: {slug}] Located in agents/public/ but has visibility='{vis}'")
        elif "/agents/internal/" in str(fpath) and vis not in ("internal", "private"):
            errors.append(f"[Agent: {slug}] Located in agents/internal/ but has visibility='{vis}'")

        # Referenced skills validation
        for skill_ref in a.get("skills", []):
            if skill_ref not in skills:
                errors.append(
                    f"[Agent: {slug}] References non-existent skill '{skill_ref}' in frontmatter 'skills:'"
                )

        if not a.get("system_prompt", "").strip():
            warnings.append(f"[Agent: {slug}] System prompt body is empty in {fpath}")

    return errors, warnings


@catalog_app.callback(invoke_without_command=True)
def default_catalog(ctx: typer.Context):
    """Show catalog summary if no sub-command is provided."""
    if ctx.invoked_subcommand is None:
        list_catalog_cmd()


@catalog_app.command(name="list")
def list_catalog_cmd(
    visibility: str = typer.Option("all", "--visibility", "-v", help="Filter by visibility (all, public, internal)"),
):
    """List all local Markdown skills and agent profiles."""
    root = _find_monorepo_root()
    skills = load_local_skills_catalog(root, visibility_filter=visibility)
    agents = load_local_agents_catalog(root, visibility_filter=visibility)

    console.print(f"\n[bold cyan]Valstorm Markdown Catalog[/bold cyan] (Filter: [bold]{visibility}[/bold])\n")

    # Agents Table
    agent_table = Table(title=f"AI Agent Profiles ({len(agents)} total)")
    agent_table.add_column("Slug", style="bold cyan", no_wrap=True)
    agent_table.add_column("Name", style="white")
    agent_table.add_column("Visibility", style="green", no_wrap=True)
    agent_table.add_column("Tier", style="yellow", no_wrap=True)
    agent_table.add_column("Model", style="magenta")
    agent_table.add_column("Tools", style="dim", justify="right")
    agent_table.add_column("Skills", style="blue", justify="right")

    for slug, a in sorted(agents.items()):
        vis_style = "[green]public[/green]" if a["visibility"] == "public" else "[yellow]internal[/yellow]"
        agent_table.add_row(
            slug,
            a["name"],
            vis_style,
            a["model_tier"],
            a["model"],
            str(len(a.get("allowed_tools", []))),
            str(len(a.get("skills", []))),
        )
    console.print(agent_table)

    # Skills Table
    skill_table = Table(title=f"AI Procedural Skills ({len(skills)} total)")
    skill_table.add_column("Category", style="cyan", no_wrap=True)
    skill_table.add_column("Slug", style="bold white", no_wrap=True)
    skill_table.add_column("Name", style="green")
    skill_table.add_column("Visibility", style="yellow", no_wrap=True)
    skill_table.add_column("Description", style="dim")

    for slug, s in sorted(skills.items(), key=lambda x: (x[1]["category"], x[0])):
        desc = (s.get("description") or "").replace("\n", " ")
        if len(desc) > 55:
            desc = desc[:52] + "..."
        vis_style = "[green]public[/green]" if s["visibility"] == "public" else "[yellow]internal[/yellow]"
        skill_table.add_row(s["category"], slug, s["name"], vis_style, desc)

    console.print("\n")
    console.print(skill_table)


@catalog_app.command(name="validate")
@catalog_app.command(name="lint")
def validate_catalog_cmd():
    """Validate and lint YAML frontmatter and references across all Markdown catalog files."""
    root = _find_monorepo_root()
    console.print(f"[dim]Scanning catalog at {root}...[/dim]")
    errors, warnings = validate_catalog_integrity(root)

    if warnings:
        console.print(f"\n[bold yellow]⚠️  Warnings ({len(warnings)}):[/bold yellow]")
        for w in warnings:
            console.print(f"  • {w}")

    if errors:
        console.print(f"\n[bold red]❌ Validation Failed ({len(errors)} errors):[/bold red]")
        for e in errors:
            console.print(f"  • {e}")
        raise typer.Exit(1)

    console.print("\n[bold green]✅ All frontmatter schemas, visibility segregations, and skill references are VALID![/bold green]\n")


@catalog_app.command(name="build-profiles")
def build_profiles_cmd(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Target directory for compiled profiles (default: ~/.valstorm/profiles)",
    ),
):
    """Compile local Markdown agent profiles and skills into ~/.valstorm/profiles/*.json for local execution."""
    root = _find_monorepo_root()
    errors, warnings = validate_catalog_integrity(root)
    if errors:
        console.print(f"[bold red]Cannot compile profiles due to {len(errors)} validation errors:[/bold red]")
        for e in errors:
            console.print(f"  • {e}")
        raise typer.Exit(1)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (Path.home() / ".valstorm" / "profiles")
    target_dir.mkdir(parents=True, exist_ok=True)

    target_skills_dir = Path.home() / ".valstorm" / "skills"
    target_skills_dir.mkdir(parents=True, exist_ok=True)

    skills = load_local_skills_catalog(root, visibility_filter="all")
    agents = load_local_agents_catalog(root, visibility_filter="all")

    # 1. Clean up removed skills in ~/.valstorm/skills
    for s_dir in list(target_skills_dir.glob("*/*")):
        if s_dir.is_dir() and s_dir.name not in skills:
            shutil.rmtree(s_dir, ignore_errors=True)
    for cat_dir in list(target_skills_dir.iterdir()):
        if cat_dir.is_dir() and not any(cat_dir.iterdir()):
            cat_dir.rmdir()

    # 2. Write skills to ~/.valstorm/skills/<category>/<slug>/SKILL.md
    synced_skills = 0
    for slug, s in skills.items():
        cat = s["category"]
        s_dir = target_skills_dir / cat / slug
        s_dir.mkdir(parents=True, exist_ok=True)
        s_file = s_dir / "SKILL.md"

        fm = {
            "name": s["name"],
            "slug": slug,
            "api_name": s["api_name"],
            "category": cat,
            "description": s["description"],
            "visibility": s["visibility"],
            "is_active": s["is_active"],
            "version": s["version"],
            "tags": s["tags"],
        }
        content = _dump_frontmatter(fm) + "\n" + s["body"].strip() + "\n"
        s_file.write_text(content, encoding="utf-8")
        synced_skills += 1

    # 3. Write profile JSONs to ~/.valstorm/profiles/<slug>.json
    synced_agents = 0
    for slug, a in agents.items():
        prof_doc = {
            "name": a["name"],
            "api_name": a["api_name"],
            "description": a["description"],
            "visibility": a["visibility"],
            "model_tier": a["model_tier"],
            "model": a["model"],
            "provider": a["provider"],
            "is_active": a["is_active"],
            "allowed_tools": a["allowed_tools"],
            "skills": a["skills"],
            "attached_skill_slugs": a["skills"],
            "system_prompt": a["system_prompt"],
            "tag": a.get("tag", []),
            "scoped_paths": a.get("scoped_paths", []),
        }
        out_json = target_dir / f"{slug}.json"
        out_json.write_text(json.dumps(prof_doc, indent=2), encoding="utf-8")
        synced_agents += 1

    console.print(f"\n[bold green]✅ Profiles Compiled Successfully![/bold green]")
    console.print(f"  • Synced [bold]{synced_agents}[/bold] agent profiles to: [cyan]{target_dir}[/cyan]")
    console.print(f"  • Synced [bold]{synced_skills}[/bold] skills to: [cyan]{target_skills_dir}[/cyan]\n")


async def _async_push_catalog(
    visibility: str = "all",
    dry_run: bool = False,
    env: str = "local",
    token: Optional[str] = None,
    valstorm_internal_org: str = VALSTORM_INTERNAL_ORG_ID,
):
    """Executes catalog push to Valstorm Cloud with org-scoped visibility."""
    root = _find_monorepo_root()
    errors, warnings = validate_catalog_integrity(root)
    if errors:
        console.print(f"[bold red]Cannot push catalog due to {len(errors)} validation errors:[/bold red]")
        for e in errors:
            console.print(f"  • {e}")
        raise typer.Exit(1)

    local_skills = load_local_skills_catalog(root, visibility_filter=visibility)
    local_agents = load_local_agents_catalog(root, visibility_filter=visibility)

    console.print(
        f"\n[bold cyan]Valstorm Cloud Catalog Push[/bold cyan] (Environment: [bold]{env}[/bold], Visibility: [bold]{visibility}[/bold], Dry Run: [bold]{dry_run}[/bold], Internal Org: [bold]{valstorm_internal_org}[/bold])"
    )

    client = ValstormApiClient(env=env, token=token)

    # 1. Fetch Remote Skills
    remote_skill_map: Dict[str, Dict[str, Any]] = {}
    try:
        res = await client.sql_query("SELECT id, name, api_name, category, visibility FROM ai_skill")
        records = res if isinstance(res, list) else res.get("records", [])
        for r in records:
            if r.get("api_name"):
                remote_skill_map[r["api_name"].lower().strip()] = r
    except Exception as e:
        console.print(f"[yellow]⚠️ Notice: Could not query remote ai_skill collection ({e}). Proceeding in simulated mode.[/yellow]")

    # 2. Fetch Remote Agents
    remote_agent_map: Dict[str, Dict[str, Any]] = {}
    try:
        res_agents = await client.sql_query("SELECT id, name, api_name, visibility FROM ai_agent")
        records_a = res_agents if isinstance(res_agents, list) else res_agents.get("records", [])
        for r in records_a:
            if r.get("api_name"):
                remote_agent_map[r["api_name"].lower().strip()] = r
    except Exception:
        pass

    # Check for deleted remote skills to purge
    local_valid_slugs = {s["api_name"].lower() for s in local_skills.values()} | {s["slug"].lower() for s in local_skills.values()}
    skills_to_delete_ids = []
    for r_slug, r_rec in remote_skill_map.items():
        if r_slug not in local_valid_slugs and r_rec.get("id"):
            skills_to_delete_ids.append(r_rec["id"])

    # Check for deleted remote agents to purge
    local_valid_agent_slugs = {a["api_name"].lower() for a in local_agents.values()} | {a["slug"].lower() for a in local_agents.values()}
    agents_to_delete_ids = []
    for r_slug, r_rec in remote_agent_map.items():
        if r_slug not in local_valid_agent_slugs and r_rec.get("id"):
            agents_to_delete_ids.append(r_rec["id"])

    skills_to_create = []
    skills_to_update = []
    for slug, s in local_skills.items():
        doc = {
            "name": s["name"],
            "api_name": s["api_name"],
            "category": s["category"],
            "description": s["description"],
            "body": s["body"],
            "visibility": s["visibility"],
            "is_active": s["is_active"],
            "version": s["version"],
        }
        match = remote_skill_map.get(s["api_name"].lower().strip()) or remote_skill_map.get(slug.lower())
        if match and match.get("id"):
            doc["id"] = match["id"]
            skills_to_update.append(doc)
        else:
            skills_to_create.append(doc)

    agents_to_create = []
    agents_to_update = []
    for slug, a in local_agents.items():
        resolved_skill_ids = []
        for s_ref in a["skills"]:
            remote_s = remote_skill_map.get(s_ref.lower()) or remote_skill_map.get(s_ref.replace("-", "_").lower())
            if remote_s and remote_s.get("id"):
                resolved_skill_ids.append(remote_s["id"])

        prov = (a.get("provider") or "Gemini").strip()
        if prov.lower() == "gemini":
            prov = "Gemini"
        elif prov.lower() in ("claude", "anthropic"):
            prov = "Claude"
        elif prov.lower() in ("chatgpt", "openai", "gpt"):
            prov = "OpenAI"

        agent_doc = {
            "name": a["name"],
            "api_name": a["api_name"],
            "description": a["description"],
            "model_tier": a["model_tier"],
            "model": a["model"],
            "provider": prov,
            "system_prompt": a["system_prompt"],
            "allowed_tools": a["allowed_tools"],
            "ai_skills": resolved_skill_ids,
            "tag": a.get("tag", []),
            "scoped_paths": a.get("scoped_paths", []),
            "is_active": a["is_active"],
        }
        match_a = remote_agent_map.get(a["api_name"].lower().strip()) or remote_agent_map.get(slug.lower())
        if match_a and match_a.get("id"):
            agent_doc["id"] = match_a["id"]
            agents_to_update.append(agent_doc)
        else:
            agents_to_create.append(agent_doc)

    # Output Summary Table
    summary_table = Table(title="Push Execution Plan")
    summary_table.add_column("Resource", style="bold cyan")
    summary_table.add_column("To Create", style="green")
    summary_table.add_column("To Update", style="yellow")
    summary_table.add_column("To Delete (Purge)", style="red")
    summary_table.add_column("Total Local", style="white")

    summary_table.add_row(
        "Skills (ai_skill)",
        str(len(skills_to_create)),
        str(len(skills_to_update)),
        str(len(skills_to_delete_ids)),
        str(len(local_skills)),
    )
    summary_table.add_row(
        "Agents (ai_agent)",
        str(len(agents_to_create)),
        str(len(agents_to_update)),
        str(len(agents_to_delete_ids)),
        str(len(local_agents)),
    )
    console.print(summary_table)

    if dry_run:
        console.print("\n[bold yellow]🔍 DRY RUN COMPLETE:[/bold yellow] No remote mutations performed.\n")
        return

    # Execute Mutations
    try:
        if skills_to_delete_ids:
            console.print(f"Purging {len(skills_to_delete_ids)} deprecated skills in Cloud...")
            await client.records_delete("ai_skill", skills_to_delete_ids)

        if agents_to_delete_ids:
            console.print(f"Purging {len(agents_to_delete_ids)} pruned agents in Cloud...")
            await client.records_delete("ai_agent", agents_to_delete_ids)

        if skills_to_create:
            console.print(f"Creating {len(skills_to_create)} new skills in Cloud...")
            await client.records_create("ai_skill", skills_to_create)
        if skills_to_update:
            console.print(f"Updating {len(skills_to_update)} existing skills in Cloud...")
            await client.records_update("ai_skill", skills_to_update)

        if agents_to_create:
            console.print(f"Creating {len(agents_to_create)} new agents in Cloud...")
            await client.records_create("ai_agent", agents_to_create)
        if agents_to_update:
            console.print(f"Updating {len(agents_to_update)} existing agents in Cloud...")
            await client.records_update("ai_agent", agents_to_update)

        console.print("\n[bold green]✅ Valstorm Cloud Push Completed Successfully![/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Push Error:[/bold red] {e}")


@catalog_app.command(name="push")
def push_catalog_cmd(
    visibility: str = typer.Option("all", "--visibility", "-v", help="Filter items to push (all, public, internal)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Simulate push without modifying Cloud records"),
    env: str = typer.Option("local", "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Override auth token"),
    valstorm_internal_org: str = typer.Option(
        VALSTORM_INTERNAL_ORG_ID,
        "--valstorm-internal-org",
        help="Valstorm internal org ID for proprietary skills",
    ),
):
    """Push local Markdown skills and agent catalog to Valstorm Cloud."""
    asyncio.run(
        _async_push_catalog(
            visibility=visibility,
            dry_run=dry_run,
            env=env,
            token=token,
            valstorm_internal_org=valstorm_internal_org,
        )
    )


async def _async_pull_catalog(
    visibility: str = "all",
    env: str = "local",
    token: Optional[str] = None,
):
    """Executes catalog pull from Valstorm Cloud to local Markdown files."""
    root = _find_monorepo_root()
    skills_dir = root / "skills"
    agents_dir = root / "agents"

    console.print(f"\n[bold cyan]Valstorm Cloud Catalog Pull[/bold cyan] (Environment: [bold]{env}[/bold], Visibility: [bold]{visibility}[/bold])")

    client = ValstormApiClient(env=env, token=token)

    try:
        res_skills = await client.sql_query(
            "SELECT id, name, api_name, category, description, body, visibility, is_active, version FROM ai_skill"
        )
        remote_skills = res_skills if isinstance(res_skills, list) else res_skills.get("records", [])
        remote_skill_id_to_slug = {}

        pulled_skills = 0
        for s in remote_skills:
            slug = (s.get("api_name") or s.get("name", "")).lower().strip().replace(" ", "-").replace("_", "-")
            if s.get("id"):
                remote_skill_id_to_slug[s["id"]] = slug

            vis = s.get("visibility", "public").lower()
            if visibility != "all" and vis != visibility and not (visibility == "internal" and vis in ("internal", "private")):
                continue

            target_base = skills_dir / ("internal" if vis in ("internal", "private") else "public")
            cat = s.get("category", "general")
            cat_dir = target_base / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            out_file = cat_dir / f"{slug}.md"

            fm = {
                "name": s.get("name") or slug.replace("-", " ").title(),
                "slug": slug,
                "api_name": s.get("api_name") or slug,
                "category": cat,
                "description": s.get("description", ""),
                "visibility": vis,
                "is_active": s.get("is_active", True),
                "version": s.get("version", "1.0.0"),
            }
            body = s.get("body", "")
            content = _dump_frontmatter(fm) + "\n" + body.strip() + "\n"
            out_file.write_text(content, encoding="utf-8")
            pulled_skills += 1

        res_agents = await client.sql_query(
            "SELECT id, name, api_name, description, visibility, model_tier, model, provider, allowed_tools, ai_skills, system_prompt, tag, scoped_paths, is_active FROM ai_agent"
        )
        remote_agents = res_agents if isinstance(res_agents, list) else res_agents.get("records", [])

        pulled_agents = 0
        for a in remote_agents:
            slug = (a.get("api_name") or a.get("name", "")).lower().strip().replace(" ", "-").replace("_", "-")
            vis = a.get("visibility", "public").lower()
            if visibility != "all" and vis != visibility and not (visibility == "internal" and vis in ("internal", "private")):
                continue

            target_base = agents_dir / ("internal" if vis in ("internal", "private") else "public")
            target_base.mkdir(parents=True, exist_ok=True)
            out_file = target_base / f"{slug}.md"

            attached_slugs = []
            for s_id in a.get("ai_skills", []):
                if s_id in remote_skill_id_to_slug:
                    attached_slugs.append(remote_skill_id_to_slug[s_id])
                else:
                    attached_slugs.append(s_id)

            fm = {
                "name": a.get("name") or slug.replace("-", " ").title(),
                "slug": slug,
                "api_name": a.get("api_name") or slug,
                "description": a.get("description", ""),
                "visibility": vis,
                "model_tier": a.get("model_tier", "tier_2"),
                "model": a.get("model", "gemini-flash-latest"),
                "provider": a.get("provider", "gemini"),
                "is_active": a.get("is_active", True),
                "allowed_tools": a.get("allowed_tools", []),
                "skills": attached_slugs,
                "tag": a.get("tag", []),
                "scoped_paths": a.get("scoped_paths", []),
            }
            prompt = a.get("system_prompt", "")
            content = _dump_frontmatter(fm) + "\n" + prompt.strip() + "\n"
            out_file.write_text(content, encoding="utf-8")
            pulled_agents += 1

        console.print(f"\n[bold green]✅ Valstorm Cloud Pull Complete![/bold green]")
        console.print(f"  • Pulled [bold]{pulled_skills}[/bold] skills and [bold]{pulled_agents}[/bold] agents into local Markdown files.\n")
    except Exception as e:
        console.print(f"[bold yellow]⚠️ Notice: Could not connect to remote Valstorm Cloud endpoint ({e}). Local Markdown catalog unchanged.[/bold yellow]")


@catalog_app.command(name="pull")
def pull_catalog_cmd(
    visibility: str = typer.Option("all", "--visibility", "-v", help="Filter items to pull (all, public, internal)"),
    env: str = typer.Option("local", "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Override auth token"),
):
    """Pull remote Valstorm Cloud skills and agents into local Markdown files."""
    asyncio.run(_async_pull_catalog(visibility=visibility, env=env, token=token))


@catalog_app.command(name="cleanup")
@catalog_app.command(name="reorganize")
def cleanup_catalog_cmd(
    cloud_sync: bool = typer.Option(False, "--cloud-sync", "-c", help="Execute Valstorm Cloud sync"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Simulate Cloud changes without mutating"),
    env: str = typer.Option("local", "--env", "-e", help="Valstorm environment (local, dev, prod)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Valstorm API token override"),
    valstorm_internal_org: str = typer.Option(
        VALSTORM_INTERNAL_ORG_ID,
        "--valstorm-internal-org",
        help="Valstorm internal org ID for proprietary skills",
    ),
):
    """Execute reproducible cleanup, migrations, and re-organization of skills and agents."""
    from scripts.reorganize_skills_and_agents import reorganize
    reorganize(
        monorepo_root=_find_monorepo_root(),
        cloud_sync=cloud_sync,
        dry_run=dry_run,
        env=env,
        token=token,
        valstorm_internal_org=valstorm_internal_org,
    )
