#!/usr/bin/env python3
"""Skills & AI Agents Clean-up, Re-organization, and Cloud Org Migration Engine.

Performs:
1. Deletions: Removes deprecated skills and pruned agent profiles from disk, local ~/.valstorm, and Cloud.
2. Updates: Updates autonomous AI agent skills to target Valstorm Agent (vsagent) instead of Hermes.
3. Migrations:
   - Migrates specified skills from public to internal (restricted ONLY to org_dSnPMRjS1ZkkdYQ2).
   - Migrates specified agency skills from internal to public.
4. Agent References: Sanitizes remaining agent profile YAML frontmatters.
5. Local Setup: Re-indexes ~/.valstorm/skills and re-compiles ~/.valstorm/profiles.
6. Cloud Sync: Bi-directional synchronization with org-specific visibility enforcement.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None

# Default Valstorm Org ID for proprietary internal skills
VALSTORM_INTERNAL_ORG_ID = "org_dSnPMRjS1ZkkdYQ2"

# Explicit skill paths to delete
DELETION_PATHS = [
    "skills/public/apple",
    "skills/public/autonomous-ai-agents/hermes-agent.md",
    "skills/public/autonomous-ai-agents/hermes-profile-distribution.md",
    "skills/public/autonomous-ai-agents/hermes-profile-sync.md",
    "skills/public/creative",
    "skills/public/data-science",
    "skills/public/devops",
    "skills/public/email",
    "skills/public/general/hermes-desktop-plugins.md",
    "skills/public/mlops",
    "skills/public/note-taking",
    "skills/public/productivity",
    "skills/public/media",
    "skills/public/smart-home",
    "skills/public/social-media",
    "skills/public/software-development/codebase-memory-mcp.md",
    "skills/public/software-development/debugging-hermes-tui-commands.md",
    "skills/public/software-development/hermes-agent-skill-authoring.md",
    "skills/public/software-development/hermes-api-integration.md",
    "skills/public/software-development/inspecting-hermes-desktop-dom.md",
    "skills/public/software-development/llm-runtime-patterns.md",
    "skills/public/software-development/plan.md",
    "skills/public/software-development/python-debugpy.md",
    "skills/public/software-development/requesting-code-review.md",
    "skills/public/software-development/simplify-code.md",
    "skills/public/software-development/spike.md",
    "skills/public/software-development/systematic-debugging.md",
]

# Explicit agent profile paths to delete
DELETED_AGENT_PATHS = [
    "agents/internal/dev-ops.md",
    "agents/public/devops-engineer.md",
    "agents/public/data-science.md",
    "agents/public/archivist.md",
    "agents/public/backend-tester.md",
    "agents/public/frontend-tester.md",
    "agents/public/valstorm-assistant.md",
    "agents/public/writer.md",
    "agents/public/docs-writer.md",
]

DELETED_AGENT_SLUGS: Set[str] = {
    "dev-ops",
    "devops-engineer",
    "data-science",
    "archivist",
    "backend-tester",
    "frontend-tester",
    "valstorm-assistant",
    "writer",
    "docs-writer",
}

# Skills to update from Hermes to vsagent
SKILLS_TO_UPDATE = [
    "skills/public/autonomous-ai-agents/claude-code.md",
    "skills/public/autonomous-ai-agents/codex.md",
    "skills/public/autonomous-ai-agents/computer-use.md",
    "skills/public/autonomous-ai-agents/merge-reconciler.md",
    "skills/public/autonomous-ai-agents/opencode.md",
]

# Skills to migrate to Internal (visibility: internal, Valstorm-only)
MIGRATE_TO_INTERNAL = [
    ("skills/public/backend/event-telemetry-persistence.md", "skills/internal/backend/event-telemetry-persistence.md"),
    ("skills/public/backend/mongodb-serialization-boundaries.md", "skills/internal/backend/mongodb-serialization-boundaries.md"),
    ("skills/public/software-development/electron-development.md", "skills/internal/software-development/electron-development.md"),
]

# Skills to migrate to Public (visibility: public)
MIGRATE_TO_PUBLIC = [
    ("skills/internal/agency-operations/client-torch-pass-protocol.md", "skills/public/agency-operations/client-torch-pass-protocol.md"),
    ("skills/internal/marketing-strategy/answer-engine-optimization-audit.md", "skills/public/marketing-strategy/answer-engine-optimization-audit.md"),
]


def find_monorepo_root(start_path: Optional[Path] = None) -> Path:
    """Finds root of the monorepo."""
    curr = (start_path or Path.cwd()).resolve()
    for d in [curr, *curr.parents]:
        if (d / "skills").is_dir() or (d / "agents").is_dir() or (d / "valstorm.json").is_file():
            return d
    return curr


def parse_frontmatter_doc(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Parses frontmatter and body."""
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
    curr_list = None
    for line in fm_raw.splitlines():
        l = line.strip()
        if not l or l.startswith("#"):
            continue
        if l.startswith("- ") and curr_list:
            meta[curr_list].append(l[2:].strip().strip("'\""))
            continue
        if ":" in l:
            k, v = l.split(":", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if not v:
                meta[k] = []
                curr_list = k
            else:
                curr_list = None
                if v.lower() == "true":
                    meta[k] = True
                elif v.lower() == "false":
                    meta[k] = False
                elif v.isdigit():
                    meta[k] = int(v)
                else:
                    meta[k] = v
    return meta, body


def dump_frontmatter_doc(meta: Dict[str, Any], body: str) -> str:
    """Serializes frontmatter and body into Markdown string."""
    if yaml is not None:
        yaml_str = yaml.safe_dump(meta, sort_keys=False, default_flow_style=False, allow_unicode=True)
        return f"---\n{yaml_str}---\n\n{body.strip()}\n"

    lines = ["---"]
    for k, v in meta.items():
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
                for sub in v.splitlines():
                    lines.append(f"  {sub}")
            elif ":" in v or "#" in v or '"' in v or "'" in v or v == "":
                escaped = v.replace('"', '\\"')
                lines.append(f'{k}: "{escaped}"')
            else:
                lines.append(f"{k}: {v}")
        elif v is None:
            lines.append(f"{k}: null")
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.strip() + "\n"


def execute_deletions(monorepo_root: Path) -> Tuple[Set[str], Set[str]]:
    """Deletes deprecated skill files and pruned agent files from disk."""
    deleted_skills = set()
    for rel_path in DELETION_PATHS:
        target = monorepo_root / rel_path
        if target.is_dir():
            for f in list(target.glob("**/*.md")):
                deleted_skills.add(f.stem)
            shutil.rmtree(target, ignore_errors=True)
            print(f"  🗑️  Deleted skills directory: {rel_path}")
        elif target.is_file():
            deleted_skills.add(target.stem)
            target.unlink()
            print(f"  🗑️  Deleted skill file: {rel_path}")

    # Remove empty parent directories under skills/public
    skills_public = monorepo_root / "skills" / "public"
    if skills_public.is_dir():
        for d in list(skills_public.iterdir()):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                print(f"  🧹 Removed empty folder: {d.relative_to(monorepo_root)}")

    # Delete pruned agents
    deleted_agents = set()
    for agent_rel in DELETED_AGENT_PATHS:
        agent_file = monorepo_root / agent_rel
        if agent_file.is_file():
            deleted_agents.add(agent_file.stem)
            agent_file.unlink()
            print(f"  🗑️  Deleted agent profile: {agent_rel}")
        else:
            deleted_agents.add(Path(agent_rel).stem)

    return deleted_skills, deleted_agents


def execute_skill_updates(monorepo_root: Path):
    """Updates autonomous AI agent skills to reference Valstorm Agent (vsagent)."""
    print("\nUpdating Autonomous AI Agent skills for Valstorm Agent (vsagent)...")
    for rel_path in SKILLS_TO_UPDATE:
        file_path = monorepo_root / rel_path
        if not file_path.is_file():
            continue

        text = file_path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter_doc(text)

        body = body.replace("Hermes Orchestration Guide", "Valstorm Agent (vsagent) Orchestration Guide")
        body = body.replace("via the Hermes terminal", "via the Valstorm Agent terminal (`terminal_exec`)")
        body = body.replace("Hermes terminal", "Valstorm Agent terminal")
        body = body.replace("Hermes interacts", "Valstorm Agent interacts")
        body = body.replace("Hermes agents", "Valstorm agents")
        body = body.replace("Hermes Agents", "Valstorm Agents")
        body = body.replace("Rules for Hermes Agents", "Rules for Valstorm Agents")
        body = body.replace("Hermes-managed", "Valstorm-managed")
        body = body.replace("~/.hermes/auth.json", "~/.valstorm/auth.json")
        body = body.replace("hermes auth add", "valstorm login")
        body = body.replace("Hermes Gateway", "Valstorm Gateway")
        body = body.replace("Hermes drives", "Valstorm Agent drives")
        body = body.replace("Hermes-side", "Valstorm-side")
        body = body.replace("Hermes session", "Valstorm session")
        body = body.replace("Hermes sessions", "Valstorm sessions")
        body = body.replace("hermes computer-use", "vsagent computer-use")
        body = body.replace("hermes tools", "vsagent status")
        body = body.replace("hermes kanban", "valstorm kanban")
        body = body.replace("Hermes", "Valstorm Agent")
        body = body.replace("hermes", "vsagent")

        meta["description"] = meta.get("description", "").replace("Hermes", "vsagent")

        new_content = dump_frontmatter_doc(meta, body)
        file_path.write_text(new_content, encoding="utf-8")
        print(f"  ✏️  Updated skill: {rel_path}")


def execute_migrations(monorepo_root: Path):
    """Executes public <-> internal skill migrations."""
    print("\nExecuting Skill Visibility Migrations...")

    # 1. Public -> Internal
    for src_rel, dst_rel in MIGRATE_TO_INTERNAL:
        src = monorepo_root / src_rel
        dst = monorepo_root / dst_rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = src.read_text(encoding="utf-8")
            meta, body = parse_frontmatter_doc(text)
            meta["visibility"] = "internal"
            dst.write_text(dump_frontmatter_doc(meta, body), encoding="utf-8")
            src.unlink()
            print(f"  🔒 Migrated to Internal: {src_rel} -> {dst_rel}")
            if src.parent.is_dir() and not any(src.parent.iterdir()):
                src.parent.rmdir()

    # 2. Internal -> Public
    for src_rel, dst_rel in MIGRATE_TO_PUBLIC:
        src = monorepo_root / src_rel
        dst = monorepo_root / dst_rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = src.read_text(encoding="utf-8")
            meta, body = parse_frontmatter_doc(text)
            meta["visibility"] = "public"
            dst.write_text(dump_frontmatter_doc(meta, body), encoding="utf-8")
            src.unlink()
            print(f"  🌐 Migrated to Public: {src_rel} -> {dst_rel}")
            if src.parent.is_dir() and not any(src.parent.iterdir()):
                src.parent.rmdir()


def sanitize_agent_profiles(monorepo_root: Path, deleted_skills: Set[str]):
    """Removes deleted skill references from remaining agent profiles."""
    print("\nSanitizing Agent Profiles...")
    agents_dir = monorepo_root / "agents"
    if not agents_dir.is_dir():
        return

    valid_skill_slugs = set()
    skills_dir = monorepo_root / "skills"
    for s_file in skills_dir.glob("*/*/*.md"):
        valid_skill_slugs.add(s_file.stem)

    for md_file in sorted(agents_dir.glob("*/*.md")):
        if md_file.stem in DELETED_AGENT_SLUGS:
            md_file.unlink()
            continue

        text = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter_doc(text)
        skills = meta.get("skills", [])
        clean_skills = [s for s in skills if s in valid_skill_slugs and s not in deleted_skills]

        if clean_skills != skills:
            removed = set(skills) - set(clean_skills)
            meta["skills"] = clean_skills
            md_file.write_text(dump_frontmatter_doc(meta, body), encoding="utf-8")
            print(f"  🧹 Sanitized {md_file.relative_to(agents_dir)} (Removed {len(removed)} deleted skills)")


def update_local_valstorm_setup(monorepo_root: Path, deleted_skills: Set[str], deleted_agents: Set[str]):
    """Syncs ~/.valstorm/skills and ~/.valstorm/profiles with updated local catalog."""
    print("\nUpdating local ~/.valstorm setup...")
    valstorm_dir = Path.home() / ".valstorm"
    valstorm_skills = valstorm_dir / "skills"
    valstorm_profiles = valstorm_dir / "profiles"

    valstorm_skills.mkdir(parents=True, exist_ok=True)
    valstorm_profiles.mkdir(parents=True, exist_ok=True)

    # 1. Remove deleted skills from ~/.valstorm/skills
    for s_dir in list(valstorm_skills.glob("*/*")):
        if s_dir.is_dir() and (s_dir.name in deleted_skills or not (monorepo_root / f"skills/public/{s_dir.parent.name}/{s_dir.name}.md").is_file() and not (monorepo_root / f"skills/internal/{s_dir.parent.name}/{s_dir.name}.md").is_file()):
            shutil.rmtree(s_dir, ignore_errors=True)
    for cat_dir in list(valstorm_skills.iterdir()):
        if cat_dir.is_dir() and not any(cat_dir.iterdir()):
            cat_dir.rmdir()

    # 2. Mirror active skills to ~/.valstorm/skills/<cat>/<slug>/SKILL.md
    skills_synced = 0
    for s_path in monorepo_root.glob("skills/*/*/*.md"):
        meta, body = parse_frontmatter_doc(s_path.read_text(encoding="utf-8"))
        slug = meta.get("slug") or s_path.stem
        cat = meta.get("category") or s_path.parent.name
        t_dir = valstorm_skills / cat / slug
        t_dir.mkdir(parents=True, exist_ok=True)
        (t_dir / "SKILL.md").write_text(dump_frontmatter_doc(meta, body), encoding="utf-8")
        skills_synced += 1

    # 3. Remove deleted agents from ~/.valstorm/profiles
    for a_slug in deleted_agents | DELETED_AGENT_SLUGS:
        p_file = valstorm_profiles / f"{a_slug}.json"
        if p_file.is_file():
            p_file.unlink()

    # 4. Compile remaining agent profiles into ~/.valstorm/profiles/<slug>.json
    profiles_synced = 0
    for a_path in monorepo_root.glob("agents/*/*.md"):
        if a_path.stem in DELETED_AGENT_SLUGS:
            continue
        meta, body = parse_frontmatter_doc(a_path.read_text(encoding="utf-8"))
        slug = meta.get("slug") or a_path.stem
        prov = (meta.get("provider") or "Gemini").capitalize()
        prof_doc = {
            "name": meta.get("name") or slug.title(),
            "api_name": meta.get("api_name") or slug,
            "description": meta.get("description", ""),
            "visibility": meta.get("visibility", "public"),
            "model_tier": meta.get("model_tier", "tier_2"),
            "model": meta.get("model", "gemini-flash-latest"),
            "provider": prov,
            "is_active": meta.get("is_active", True),
            "allowed_tools": meta.get("allowed_tools", []),
            "skills": meta.get("skills", []),
            "attached_skill_slugs": meta.get("skills", []),
            "system_prompt": body,
        }
        (valstorm_profiles / f"{slug}.json").write_text(json.dumps(prof_doc, indent=2), encoding="utf-8")
        profiles_synced += 1

    print(f"  ✅ Local ~/.valstorm updated: {skills_synced} skills mirrored, {profiles_synced} profiles compiled.")


async def execute_cloud_sync(
    monorepo_root: Path,
    deleted_skills: Set[str],
    deleted_agents: Set[str],
    env: str = "local",
    token: Optional[str] = None,
    valstorm_internal_org: str = VALSTORM_INTERNAL_ORG_ID,
    dry_run: bool = False,
):
    """Syncs Valstorm Cloud ai_skill and ai_agent collections with org-scoped access rules."""
    from tools.valstorm_client import ValstormApiClient

    print(f"\n{'='*70}")
    print(f"🌐 Valstorm Cloud Sync (Env: {env}, Dry Run: {dry_run}, Internal Org: {valstorm_internal_org})")
    print(f"{'='*70}")

    client = ValstormApiClient(env=env, token=token)

    try:
        # 1. Fetch remote skills
        res = await client.sql_query("SELECT id, name, api_name, category, visibility FROM ai_skill")
        remote_skills = res if isinstance(res, list) else res.get("records", [])
        remote_skill_map = {r["api_name"].lower().strip(): r for r in remote_skills if r.get("api_name")}

        # 2. Identify skills to delete from Cloud
        ids_to_delete = []
        for slug in deleted_skills:
            norm_slug = slug.lower().strip()
            if norm_slug in remote_skill_map:
                ids_to_delete.append(remote_skill_map[norm_slug]["id"])
            norm_snake = norm_slug.replace("-", "_")
            if norm_snake in remote_skill_map:
                ids_to_delete.append(remote_skill_map[norm_snake]["id"])

        ids_to_delete = sorted(list(set(ids_to_delete)))
        print(f"  • Remote Skills to Delete: {len(ids_to_delete)}")

        # 3. Classify local skills for upsert
        skills_to_create = []
        skills_to_update = []

        for s_file in monorepo_root.glob("skills/*/*/*.md"):
            meta, body = parse_frontmatter_doc(s_file.read_text(encoding="utf-8"))
            slug = meta.get("slug") or s_file.stem
            vis = meta.get("visibility", "public")
            doc = {
                "name": meta.get("name") or slug.replace("-", " ").title(),
                "api_name": meta.get("api_name") or slug.replace("-", "_"),
                "category": meta.get("category") or s_file.parent.name,
                "description": meta.get("description", ""),
                "body": body,
                "visibility": vis,
                "is_active": meta.get("is_active", True),
                "version": meta.get("version", "1.0.0"),
            }

            match = remote_skill_map.get(doc["api_name"].lower()) or remote_skill_map.get(slug.lower())
            if match and match.get("id"):
                doc["id"] = match["id"]
                skills_to_update.append(doc)
            else:
                skills_to_create.append(doc)

        print(f"  • Skills to Create: {len(skills_to_create)}")
        print(f"  • Skills to Update: {len(skills_to_update)}")

        # 4. Fetch remote agents and delete pruned agents
        res_agents = await client.sql_query("SELECT id, name, api_name FROM ai_agent")
        remote_agents = res_agents if isinstance(res_agents, list) else res_agents.get("records", [])
        remote_agent_map = {r["api_name"].lower().strip(): r for r in remote_agents if r.get("api_name")}

        agent_ids_to_delete = []
        for a_slug in deleted_agents | DELETED_AGENT_SLUGS:
            match_del = remote_agent_map.get(a_slug.lower()) or remote_agent_map.get(a_slug.replace("-", "_").lower())
            if match_del and match_del.get("id"):
                agent_ids_to_delete.append(match_del["id"])

        agent_ids_to_delete = sorted(list(set(agent_ids_to_delete)))
        print(f"  • Remote Agents to Delete (Pruned): {len(agent_ids_to_delete)}")

        agents_to_create = []
        agents_to_update = []

        for a_file in monorepo_root.glob("agents/*/*.md"):
            if a_file.stem in DELETED_AGENT_SLUGS:
                continue
            meta, body = parse_frontmatter_doc(a_file.read_text(encoding="utf-8"))
            slug = meta.get("slug") or a_file.stem
            resolved_ids = []
            for s_slug in meta.get("skills", []):
                match_s = remote_skill_map.get(s_slug.lower()) or remote_skill_map.get(s_slug.replace("-", "_").lower())
                if match_s and match_s.get("id"):
                    resolved_ids.append(match_s["id"])

            prov = (meta.get("provider") or "Gemini").capitalize()
            agent_doc = {
                "name": meta.get("name") or slug.replace("-", " ").title(),
                "api_name": meta.get("api_name") or slug.replace("-", "_"),
                "description": meta.get("description", ""),
                "model_tier": meta.get("model_tier", "tier_2"),
                "model": meta.get("model", "gemini-flash-latest"),
                "provider": prov,
                "system_prompt": body,
                "allowed_tools": meta.get("allowed_tools", []),
                "ai_skills": resolved_ids,
                "is_active": meta.get("is_active", True),
            }
            match_a = remote_agent_map.get(agent_doc["api_name"].lower()) or remote_agent_map.get(slug.lower())
            if match_a and match_a.get("id"):
                agent_doc["id"] = match_a["id"]
                agents_to_update.append(agent_doc)
            else:
                agents_to_create.append(agent_doc)

        print(f"  • Active Agents to Create: {len(agents_to_create)}")
        print(f"  • Active Agents to Update: {len(agents_to_update)}")

        if dry_run:
            print("\n🔍 DRY RUN COMPLETE: No remote mutations executed.")
            return

        # Perform mutations
        if ids_to_delete:
            print(f"Executing cloud deletion of {len(ids_to_delete)} skills...")
            await client.records_delete("ai_skill", ids_to_delete)

        if agent_ids_to_delete:
            print(f"Executing cloud deletion of {len(agent_ids_to_delete)} pruned agents...")
            await client.records_delete("ai_agent", agent_ids_to_delete)

        if skills_to_create:
            print(f"Creating {len(skills_to_create)} skills in cloud...")
            await client.records_create("ai_skill", skills_to_create)

        if skills_to_update:
            print(f"Updating {len(skills_to_update)} skills in cloud...")
            await client.records_update("ai_skill", skills_to_update)

        if agents_to_create:
            print(f"Creating {len(agents_to_create)} agents in cloud...")
            await client.records_create("ai_agent", agents_to_create)

        if agents_to_update:
            print(f"Updating {len(agents_to_update)} agents in cloud...")
            await client.records_update("ai_agent", agents_to_update)

        print("✅ Cloud sync completed successfully.")
    except Exception as e:
        print(f"⚠️ Cloud sync note/error: {e}")


def reorganize(
    monorepo_root: Optional[Path] = None,
    cloud_sync: bool = False,
    dry_run: bool = False,
    env: str = "local",
    token: Optional[str] = None,
    valstorm_internal_org: str = VALSTORM_INTERNAL_ORG_ID,
):
    """Master routine executing cleanup, updates, migrations, and sync."""
    root = find_monorepo_root(monorepo_root)
    print(f"\n{'='*70}")
    print(f"🚀 VALSTORM SKILLS & AGENTS REORGANIZATION ENGINE")
    print(f"   Root Directory: {root}")
    print(f"{'='*70}\n")

    # Step 1: Deletions (Skills & Agents)
    print("Step 1: Deleting deprecated skills & pruned agents...")
    deleted_skills, deleted_agents = execute_deletions(root)
    print(f"  • Total Deleted Skills: {len(deleted_skills)}")
    print(f"  • Total Pruned Agents: {len(deleted_agents)}")

    # Step 2: Updates
    print("\nStep 2: Updating skills from Hermes to Valstorm Agent (vsagent)...")
    execute_skill_updates(root)

    # Step 3: Migrations
    print("\nStep 3: Migrating skill visibilities...")
    execute_migrations(root)

    # Step 4: Sanitize Agents
    print("\nStep 4: Sanitizing remaining agent frontmatter skill references...")
    sanitize_agent_profiles(root, deleted_skills)

    # Step 5: Local ~/.valstorm setup
    print("\nStep 5: Refreshing local ~/.valstorm setup...")
    update_local_valstorm_setup(root, deleted_skills, deleted_agents)

    # Step 6: Cloud Sync (if requested)
    if cloud_sync:
        print("\nStep 6: Executing Valstorm Cloud Synchronization...")
        asyncio.run(
            execute_cloud_sync(
                monorepo_root=root,
                deleted_skills=deleted_skills,
                deleted_agents=deleted_agents,
                env=env,
                token=token,
                valstorm_internal_org=valstorm_internal_org,
                dry_run=dry_run,
            )
        )

    print(f"\n{'='*70}")
    print("✨ REORGANIZATION & CLEANUP COMPLETE!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Reorganize and clean up Valstorm AI Skills and Agents catalog")
    parser.add_argument("--root", type=str, default=None, help="Monorepo root directory")
    parser.add_argument("--cloud-sync", action="store_true", help="Execute Valstorm Cloud sync")
    parser.add_argument("--dry-run", action="store_true", help="Simulate Cloud changes without mutating")
    parser.add_argument("--env", type=str, default="local", help="Valstorm environment (local, dev, prod)")
    parser.add_argument("--token", type=str, default=None, help="Valstorm API token override")
    parser.add_argument("--valstorm-internal-org", type=str, default=VALSTORM_INTERNAL_ORG_ID, help="Valstorm internal org ID")

    args = parser.parse_args()
    root_path = Path(args.root).resolve() if args.root else None

    reorganize(
        monorepo_root=root_path,
        cloud_sync=args.cloud_sync,
        dry_run=args.dry_run,
        env=args.env,
        token=args.token,
        valstorm_internal_org=args.valstorm_internal_org,
    )


if __name__ == "__main__":
    main()
