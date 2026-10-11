"""Procedural Skill Discovery and Inspection Tools for Valstorm Agent Runtime.

Provides:
- skill_view: Loads, parses, and windows specific SKILL.md procedures (and sub-reference docs)
  from ~/.valstorm/skills/<category>/<slug>/SKILL.md.
- skill_list: Discovers, categorizes, and searches all installed agent skills.
"""

import asyncio
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from core.tools import ToolRegistry, tool
from tools.developer_tools import _truncate_output


def _resolve_skills_dir() -> Path:
    """Resolves local skills directory from VALSTORM_SKILLS_DIR, workspace skills/, or default ~/.valstorm/skills."""
    custom = os.environ.get("VALSTORM_SKILLS_DIR")
    if custom and custom.strip():
        p = Path(custom).expanduser().resolve()
        if p.is_dir():
            return p
    # Check workspace skills dir
    curr = Path.cwd().resolve()
    for directory in [curr, *curr.parents]:
        ws_skills = directory / "skills"
        if ws_skills.is_dir() and ((ws_skills / "public").is_dir() or (ws_skills / "internal").is_dir()):
            return ws_skills
    return (Path.home() / ".valstorm" / "skills").resolve()


def _parse_yaml_lines(lines: List[str]) -> Dict[str, Any]:
    """Parses nested YAML frontmatter lines into a Python dictionary."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any, Optional[str]]] = [(-1, root, None)]  # (indent, container, key_in_parent)

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        parent_indent, parent_container, parent_key = stack[-1]

        if stripped.startswith("- "):
            val_str = stripped[2:].strip().strip("'\"")
            if isinstance(parent_container, list):
                parent_container.append(val_str)
            elif isinstance(parent_container, dict) and parent_key is not None:
                grandparent = stack[-2][1] if len(stack) >= 2 else None
                if isinstance(grandparent, dict):
                    new_list = [val_str]
                    grandparent[parent_key] = new_list
                    stack[-1] = (parent_indent, new_list, parent_key)
        elif ":" in stripped:
            key, raw_val = stripped.split(":", 1)
            key = key.strip()
            val = raw_val.strip().strip("'\"")

            if val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
                if isinstance(parent_container, dict):
                    parent_container[key] = items
            elif val and val != "|":
                if isinstance(parent_container, dict):
                    parent_container[key] = val
            else:
                new_dict: Dict[str, Any] = {}
                if isinstance(parent_container, dict):
                    parent_container[key] = new_dict
                stack.append((indent, new_dict, key))

    return root


def _parse_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Extracts YAML-style frontmatter and body from markdown text without external dependencies."""
    if not raw_text.startswith("---"):
        return {}, raw_text

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text

    fm_raw = parts[1]
    body = parts[2].strip()

    metadata = _parse_yaml_lines(fm_raw.splitlines())

    # Normalize description if multiline block |
    if metadata.get("description") in ("|", "", None):
        desc_lines = []
        for bl in body.splitlines():
            if bl.startswith("#"):
                continue
            if bl.strip():
                desc_lines.append(bl.strip())
            elif desc_lines:
                break
        metadata["description"] = " ".join(desc_lines)[:250] if desc_lines else "Procedural skill guide."

    return metadata, body


def _index_all_skills(skills_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Discovers and parses metadata for all SKILL.md and Markdown skill files on disk."""
    root = skills_root or _resolve_skills_dir()
    if not root.is_dir():
        return []

    indexed_by_slug: Dict[str, Dict[str, Any]] = {}

    pattern_candidates = [
        root.glob("*/*/*.md"),
        root.glob("*/*/SKILL.md"),
        root.glob("*/*.md"),
    ]
    # Only fallback to ~/.valstorm/skills if root didn't find any skills and custom VALSTORM_SKILLS_DIR was not set
    if not os.environ.get("VALSTORM_SKILLS_DIR") and skills_root is None:
        home_skills = (Path.home() / ".valstorm" / "skills").resolve()
        if home_skills.is_dir() and home_skills != root:
            pattern_candidates.append(home_skills.glob("*/*/SKILL.md"))

    for generator in pattern_candidates:
        for skill_file in sorted(generator):
            if not skill_file.is_file():
                continue

            if skill_file.name == "SKILL.md":
                category = skill_file.parent.parent.name
                slug = skill_file.parent.name
            elif skill_file.parent.parent.name in ("public", "internal"):
                category = skill_file.parent.name
                slug = skill_file.stem
            else:
                category = skill_file.parent.name
                slug = skill_file.stem

            if slug in indexed_by_slug:
                continue

            try:
                content = skill_file.read_text(encoding="utf-8", errors="replace")
                meta, body = _parse_frontmatter(content)
            except Exception:
                meta, body = {}, ""

            title = meta.get("name") or slug
            desc = meta.get("description") or ""

            # Extract tags
            tags = []
            if isinstance(meta.get("tags"), list):
                tags = meta.get("tags", [])
            elif isinstance(meta.get("metadata"), dict) and isinstance(meta["metadata"].get("hermes"), dict):
                tags = meta["metadata"]["hermes"].get("tags", [])

            indexed_by_slug[slug] = {
                "category": category,
                "slug": slug,
                "name": title,
                "description": desc,
                "body_preview": body[:500] if body else "",
                "version": meta.get("version", "1.0.0"),
                "author": meta.get("author", "Valstorm"),
                "tags": tags,
                "path": str(skill_file),
                "directory": str(skill_file.parent),
                "size_bytes": skill_file.stat().st_size,
            }

    return sorted(list(indexed_by_slug.values()), key=lambda x: (x["category"], x["slug"]))


@tool
async def skill_list(
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Discovers and lists available agent skills, SOPs, and procedural guides.

    Use this tool to find relevant procedures for testing, monorepo migrations, electron debugging,
    code optimization, git workflows, or platform-specific task automation.

    Args:
        category: Optional category filter (e.g. 'software-development', 'research', 'devops', 'valstorm-internal').
        query: Optional search keyword to filter skills by name, description, tags, or contents.
        limit: Maximum number of skills to return (default: 50).

    Returns:
        A formatted list of matching skills with descriptions and category badges.
    """
    root = _resolve_skills_dir()
    if not root.is_dir():
        return f"No skills directory found at '{root}'."

    skills = await asyncio.to_thread(_index_all_skills, root)
    if not skills:
        return f"No skills found in '{root}'."

    filtered = skills

    # Filter by category
    if category and category.strip():
        cat_norm = category.strip().lower()
        filtered = [s for s in filtered if cat_norm in s["category"].lower()]

    # Filter by query
    if query and query.strip():
        q_norm = query.strip().lower()
        filtered = [
            s for s in filtered
            if q_norm in s["slug"].lower()
            or q_norm in str(s["name"]).lower()
            or q_norm in str(s["description"]).lower()
            or q_norm in str(s.get("body_preview", "")).lower()
            or any(q_norm in str(tag).lower() for tag in s.get("tags", []))
        ]

    total_matches = len(filtered)
    if total_matches == 0:
        cat_msg = f" in category '{category}'" if category else ""
        q_msg = f" matching query '{query}'" if query else ""
        return f"No skills found{cat_msg}{q_msg}."

    result_slice = filtered[:limit]

    output_lines = [
        f"📚 Available Agent Skills ({len(result_slice)} of {total_matches} matching):",
        "",
    ]

    for s in result_slice:
        tags_badge = f" [tags: {', '.join(s['tags'][:4])}]" if s.get("tags") else ""
        output_lines.append(f"• \033[1m{s['category']}/{s['slug']}\033[0m{tags_badge}")
        if s["description"]:
            clean_desc = s["description"].strip().replace("\n", " ")
            if len(clean_desc) > 120:
                clean_desc = clean_desc[:117] + "..."
            output_lines.append(f"  {clean_desc}")

    output_lines.append("")
    output_lines.append("Use `skill_view(name='<slug>')` to inspect the full procedure and guidelines.")

    return _truncate_output("\n".join(output_lines))


@tool
async def skill_view(
    name: str,
    category: Optional[str] = None,
    file_path: Optional[str] = None,
) -> str:
    """Reads a procedural skill document from the local skill repository.

    Use this tool when you need specialized step-by-step instructions, standard operating
    procedures, framework guidelines, or best practices for a specific task.

    Args:
        name: The skill slug/name (e.g. 'test-driven-development', 'systematic-debugging', 'python-testing').
        category: Optional category filter (e.g. 'software-development', 'valstorm-internal').
        file_path: Optional relative subfile if the skill contains supporting reference docs or scripts.

    Returns:
        The markdown text of the skill procedure or an error if not found.
    """
    clean_name = (name or "").strip().lower()
    if not clean_name:
        return "Error: Skill name or slug must be provided for skill_view."

    # Handle full category/slug format passed as name
    if "/" in clean_name and not category:
        category, clean_name = clean_name.split("/", 1)
        category = category.strip()
        clean_name = clean_name.strip()

    root = _resolve_skills_dir()
    if not root.is_dir():
        return f"Error: Skills directory not found at '{root}'."

    skills = await asyncio.to_thread(_index_all_skills, root)

    # 1. Exact match by slug
    matched_skill = None
    if category:
        cat_clean = category.strip().lower()
        for s in skills:
            if s["slug"] == clean_name and cat_clean in s["category"].lower():
                matched_skill = s
                break

    if not matched_skill:
        for s in skills:
            if s["slug"] == clean_name:
                matched_skill = s
                break

    # 2. Normalized match (replace underscores with hyphens)
    if not matched_skill:
        norm_slug = clean_name.replace("_", "-")
        for s in skills:
            if s["slug"] == norm_slug:
                matched_skill = s
                break

    # 3. Substring match fallback
    if not matched_skill:
        candidates = [s for s in skills if clean_name in s["slug"]]
        if len(candidates) == 1:
            matched_skill = candidates[0]
        elif len(candidates) > 1:
            slugs_str = ", ".join(f"{c['category']}/{c['slug']}" for c in candidates[:6])
            return (
                f"Error: Ambiguous skill name '{name}'. Multiple matches found: {slugs_str}. "
                f"Please specify exact slug or category."
            )

    if not matched_skill:
        # Suggest close matches
        close = [s["slug"] for s in skills if any(tok in s["slug"] for tok in clean_name.split("-") if len(tok) > 2)][:5]
        suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"Error: Skill '{name}' not found in '{root}'.{suggestion}\nUse `skill_list()` to view available skills."

    target_file = Path(matched_skill["path"])

    # If subfile requested
    if file_path and file_path.strip():
        sub_p = (Path(matched_skill["directory"]) / file_path.strip()).resolve()
        # Security boundary check
        if not str(sub_p).startswith(str(Path(matched_skill["directory"]).resolve())):
            return f"Error: Access denied: '{file_path}' traverses outside the skill directory."
        if not sub_p.is_file():
            return f"Error: Subfile '{file_path}' not found in skill directory '{matched_skill['slug']}'."
        target_file = sub_p

    try:
        content = await asyncio.to_thread(target_file.read_text, encoding="utf-8", errors="replace")
        header = f"=== SKILL: {matched_skill['category']}/{matched_skill['slug']} (v{matched_skill['version']}) ===\n"
        return _truncate_output(header + content)
    except Exception as e:
        return f"Error reading skill file '{target_file}': {e}"


def create_skill_tools() -> List[Any]:
    """Returns list of skill inspection and discovery tools."""
    return [skill_view, skill_list]


def register_skill_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers skill discovery and inspection tools into the provided ToolRegistry."""
    for tool_fn in create_skill_tools():
        registry.register(tool_fn)
    return registry
