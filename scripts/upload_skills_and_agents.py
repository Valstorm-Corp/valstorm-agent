#!/usr/bin/env python3
"""Stage 1 Preparation & Bundle Generator: Export Validated Skills & AI Agents JSONs (No IDs, with model_tier).

Prepares and validates local schema JSON bundles for manual upload via Valstorm CLI:
- /Users/jared/Documents/Code/monorepo/ai_skills.json  (Skills ready for 'valstorm record create ai_skill')
- /Users/jared/Documents/Code/monorepo/ai_agents.json  (Agents ready for 'valstorm record create ai_agent' with model_tier and remote skill IDs)
- ~/.valstorm/profiles/<slug>.json (Individual profile definitions for local agent runtime)

Usage:
  # Step 1: Generate ai_skills.json without IDs and sync ~/.valstorm/profiles
  uv run --project apps/agent-runtime python apps/agent-runtime/scripts/upload_skills_and_agents.py

  # Step 2: (After uploading skills and querying remote IDs): Generate ai_agents.json with live skill IDs & sync profiles
  uv run --project apps/agent-runtime python apps/agent-runtime/scripts/upload_skills_and_agents.py --skills-map remote_skills.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROFILE_SKILL_DEFAULTS = {
    "developer": [
        "bash-scripting", "code-modification-fallbacks", "codebase-inspection",
        "systematic-debugging", "test-driven-development", "valstorm-cli",
        "valstorm-backend-patterns", "valstorm-react-patterns", "simplify-code",
        "spike", "requesting-code-review", "python-debugpy", "node-inspect-debugger"
    ],
    "researcher": [
        "arxiv", "grounded-citations", "llm-wiki", "competitor-news-monitor",
        "blogwatcher", "valstorm-vfs", "vfs-search-and-rag-pipeline", "document-to-action-items"
    ]
}

PROFILE_TIER_DEFAULTS = {
    "architect": "tier_1",
    "video-editor": "tier_3",
    "archivist": "tier_3",
}

PROFILE_MODEL_DEFAULTS = {
    "tier_1": "gemini-pro-latest",
    "tier_2": "gemini-flash-latest",
    "tier_3": "gemini-flash-lite-latest",
}


def load_remote_skills_map(skills_map_path: Optional[Path]) -> Dict[str, str]:
    """Loads a slug -> id mapping from a query dump or JSON file."""
    if not skills_map_path or not skills_map_path.is_file():
        return {}

    try:
        data = json.loads(skills_map_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else (data.get("records") or data.get("data") or [])
        slug_to_id = {}
        for r in records:
            if isinstance(r, dict) and r.get("api_name") and r.get("id"):
                slug_to_id[str(r["api_name"]).strip()] = str(r["id"]).strip()
        return slug_to_id
    except Exception as e:
        print(f"⚠️ Warning: Could not parse skills map from {skills_map_path}: {e}")
        return {}


def prepare_and_validate_bundles(
    output_dir: Optional[Path] = None,
    skills_map_path: Optional[Path] = None,
    skills_source: Optional[Path] = None,
    agents_source: Optional[Path] = None,
) -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    monorepo_root = output_dir or script_dir.parent.parent.parent
    runtime_exports = script_dir.parent / "exports"
    valstorm_root = Path.home() / ".valstorm"
    valstorm_profiles_dir = valstorm_root / "profiles"
    valstorm_profiles_dir.mkdir(parents=True, exist_ok=True)

    skills_file = skills_source or runtime_exports / "ai_skills_migrated.json"
    if not skills_file.is_file():
        skills_file = valstorm_root / "migrated_skills.json"
    if not skills_file.is_file():
        skills_file = monorepo_root / "ai_skills.json"

    agents_file = agents_source or runtime_exports / "ai_agents_migrated.json"
    if not agents_file.is_file():
        agents_file = valstorm_root / "migrated_profiles.json"
    if not agents_file.is_file():
        agents_file = monorepo_root / "ai_agents.json"

    if not skills_file.is_file() or not agents_file.is_file():
        raise FileNotFoundError(
            "Source skill or agent files not found. "
            "Please run `migrate_hermes_profiles_and_skills.py` first to extract from Hermes."
        )

    raw_skills: List[Dict[str, Any]] = json.loads(skills_file.read_text(encoding="utf-8"))
    raw_agents: List[Dict[str, Any]] = json.loads(agents_file.read_text(encoding="utf-8"))

    # =========================================================================
    # 1. NORMALIZE & VALIDATE SKILLS (STRIP CLIENT IDs)
    # =========================================================================
    clean_skills: List[Dict[str, Any]] = []
    skill_slugs = set()

    for s in raw_skills:
        slug = s.get("api_name") or s.get("name", "").lower().strip().replace(" ", "-")
        skill_slugs.add(slug)

        skill_doc = {
            "name": s.get("name") or slug.replace("-", " ").title(),
            "api_name": slug,
            "category": s.get("category", "general"),
            "description": s.get("description", "")[:300],
            "body": s.get("body", ""),
            "is_active": s.get("is_active", True),
        }
        clean_skills.append(skill_doc)

    # =========================================================================
    # 2. RESOLVE REMOTE SKILL IDS & MODEL TIERS FOR AGENTS
    # =========================================================================
    remote_skill_id_map = load_remote_skills_map(skills_map_path)
    clean_agents: List[Dict[str, Any]] = []

    for a in raw_agents:
        slug = a.get("api_name") or a.get("name", "").lower().strip().replace(" ", "-")

        # Determine target skill slugs for this profile
        target_slugs = a.get("attached_skill_slugs") or PROFILE_SKILL_DEFAULTS.get(slug, [])

        # Map to remote IDs if mapping is provided
        resolved_skill_ids = []
        if remote_skill_id_map:
            for s_slug in target_slugs:
                if s_slug in remote_skill_id_map:
                    resolved_skill_ids.append(remote_skill_id_map[s_slug])

        model_tier = a.get("model_tier") or PROFILE_TIER_DEFAULTS.get(slug, "tier_2")
        model = a.get("model") or PROFILE_MODEL_DEFAULTS.get(model_tier, "gemini-flash-latest")

        agent_doc = {
            "name": a.get("name") or slug.replace("-", " ").title(),
            "api_name": slug,
            "description": a.get("description", ""),
            "model_tier": model_tier,
            "model": model,
            "provider": a.get("provider", "Gemini"),
            "system_prompt": a.get("system_prompt", ""),
            "allowed_tools": a.get("allowed_tools", []),
            "ai_skills": sorted(list(set(resolved_skill_ids))),
            "is_active": a.get("is_active", True),
        }
        clean_agents.append(agent_doc)

        # Write individual profile JSON to ~/.valstorm/profiles/<slug>.json for local runtime
        local_profile_doc = dict(agent_doc)
        local_profile_doc["attached_skill_slugs"] = target_slugs
        indiv_path = valstorm_profiles_dir / f"{slug}.json"
        indiv_path.write_text(json.dumps(local_profile_doc, indent=2), encoding="utf-8")

    # =========================================================================
    # 3. WRITE OUT LOCAL JSON FILES (WITHOUT IDs, WITH model_tier)
    # =========================================================================
    out_skills_path = monorepo_root / "ai_skills.json"
    out_agents_path = monorepo_root / "ai_agents.json"

    out_skills_path.write_text(json.dumps(clean_skills, indent=2), encoding="utf-8")
    out_agents_path.write_text(json.dumps(clean_agents, indent=2), encoding="utf-8")

    print(f"\n{'='*75}")
    print("✅ LOCAL JSON ARTIFACTS & PROFILES GENERATED SUCCESSFULLY")
    print(f"{'='*75}")
    print(f"  • Skills File: {out_skills_path} ({len(clean_skills)} skills)")
    print(f"  • Agents File: {out_agents_path} ({len(clean_agents)} agents)")
    print(f"  • Local Profiles Directory: {valstorm_profiles_dir}/ ({len(clean_agents)} profile files synced)")
    if remote_skill_id_map:
        total_linked = sum(len(a['ai_skills']) for a in clean_agents)
        print(f"  • Remote Skills Mapped: {len(remote_skill_id_map)} IDs ({total_linked} total agent skill attachments)")
    else:
        print("  • Note: 'ai_skills' on agents will be empty until --skills-map is provided with remote IDs.")
    print(f"{'='*75}\n")

    return out_skills_path, out_agents_path


def main():
    parser = argparse.ArgumentParser(description="Generate clean ai_skills.json and ai_agents.json files for Valstorm CLI (No IDs, with model_tier)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory to write JSON files (default: monorepo root)"
    )
    parser.add_argument(
        "--skills-map",
        type=str,
        default=None,
        help="Path to remote_skills.json containing created skill IDs to populate into ai_agents.json"
    )
    parser.add_argument(
        "--skills-source",
        type=str,
        default=None,
        help="Path to source skills JSON"
    )
    parser.add_argument(
        "--agents-source",
        type=str,
        default=None,
        help="Path to source agents JSON"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    skills_map = Path(args.skills_map) if args.skills_map else None
    skills_src = Path(args.skills_source) if args.skills_source else None
    agents_src = Path(args.agents_source) if args.agents_source else None

    prepare_and_validate_bundles(
        output_dir=output_dir,
        skills_map_path=skills_map,
        skills_source=skills_src,
        agents_source=agents_src
    )


if __name__ == "__main__":
    main()
