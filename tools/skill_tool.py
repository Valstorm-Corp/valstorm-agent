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
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from core.tools import ToolRegistry, tool
from tools.developer_tools import _truncate_output

SKILL_CACHE_TTL_SEC = 60.0
_TENANT_SKILLS_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def invalidate_tenant_skills_cache(org_id: Optional[str] = None) -> None:
    """Invalidates the in-memory tenant cloud skills cache (all tenants or specific org_id)."""
    if not org_id:
        _TENANT_SKILLS_CACHE.clear()
        return
    keys_to_remove = [k for k in _TENANT_SKILLS_CACHE if k.startswith(f"{org_id}:")]
    for k in keys_to_remove:
        _TENANT_SKILLS_CACHE.pop(k, None)


def _query_tenant_ai_skills(token: str, base_url: str) -> List[Dict[str, Any]]:
    """Queries active ai_skill records from the authenticated tenant's Valstorm API."""
    if not token or not base_url:
        return []
    clean_base = base_url.rstrip("/")
    if not clean_base.endswith("/v1") and "/v1" not in clean_base:
        clean_base = f"{clean_base}/v1"
    auth_header = token if str(token).startswith("Bearer ") else f"Bearer {token}"
    sql = (
        "SELECT id, name, api_name, category, description, body, is_active, created_by, modified_date "
        "FROM ai_skill WHERE is_active = true LIMIT 500"
    )
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{clean_base}/query",
                headers={"Authorization": auth_header, "Content-Type": "application/json"},
                json={"query": sql},
            )
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, list):
                    return [r for r in payload if isinstance(r, dict)]
                if isinstance(payload, dict):
                    recs = payload.get("data") or payload.get("records") or []
                    if isinstance(recs, list):
                        return [r for r in recs if isinstance(r, dict)]
    except Exception:
        pass
    return []


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


def _parse_cloud_skill_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalizes a cloud `ai_skill` database record into the standard indexed skill dictionary."""
    if not isinstance(rec, dict):
        return None
    raw_slug = rec.get("api_name") or rec.get("slug") or rec.get("name") or ""
    slug = str(raw_slug).strip().lower().replace(" ", "-")
    if not slug:
        return None

    raw_body = str(rec.get("body") or "")
    try:
        meta, parsed_body = _parse_frontmatter(raw_body)
    except Exception:
        meta, parsed_body = {}, raw_body

    category = str(rec.get("category") or meta.get("category") or "general").strip() or "general"
    title = str(rec.get("name") or meta.get("name") or slug).strip()
    desc = str(rec.get("description") or meta.get("description") or "").strip()

    tags: List[str] = []
    if isinstance(rec.get("tag"), list):
        tags = [str(t) for t in rec["tag"] if t]
    elif isinstance(rec.get("tags"), list):
        tags = [str(t) for t in rec["tags"] if t]
    elif isinstance(meta.get("tags"), list):
        tags = [str(t) for t in meta["tags"] if t]
    elif isinstance(meta.get("metadata"), dict) and isinstance(meta["metadata"].get("hermes"), dict):
        raw_tags = meta["metadata"]["hermes"].get("tags", [])
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags if t]

    ownership = str(meta.get("ownership") or ("bundled" if rec.get("created_by") in ("Valstorm Admin", "System") else "tenant"))

    return {
        "id": rec.get("id"),
        "category": category,
        "slug": slug,
        "name": title,
        "description": desc,
        "body": raw_body,
        "body_preview": (parsed_body or raw_body)[:500],
        "version": str(meta.get("version", "1.0.0")),
        "author": str(meta.get("author", "Valstorm")),
        "tags": tags,
        "ownership": ownership,
        "source": "cloud",
        "path": "",
        "directory": "",
        "size_bytes": len(raw_body.encode("utf-8")),
    }


def _get_cached_cloud_skills(force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (parsed_cloud_skills, is_cloud_mode) when a scoped Valstorm auth context is active."""
    try:
        from tools.valstorm_client import get_current_tenant_context, get_current_valstorm_auth
        scoped_auth = get_current_valstorm_auth()
        ctx = get_current_tenant_context()
    except Exception:
        return [], False

    if not scoped_auth or not scoped_auth[0] or not scoped_auth[1]:
        return [], False

    token, base_url = scoped_auth
    org_key = (ctx.org_id if (ctx and ctx.org_id) else token[:16]) or "default"
    is_cloud_mode = bool(ctx.is_cloud_mode) if ctx else False
    cache_key = f"{org_key}:{base_url}"

    now = time.monotonic()
    if not force_refresh and cache_key in _TENANT_SKILLS_CACHE:
        ts, cached_list = _TENANT_SKILLS_CACHE[cache_key]
        if (now - ts) < SKILL_CACHE_TTL_SEC:
            return cached_list, is_cloud_mode

    raw_records = _query_tenant_ai_skills(token, base_url)
    parsed_list: List[Dict[str, Any]] = []
    for rec in raw_records:
        parsed = _parse_cloud_skill_record(rec)
        if parsed:
            parsed_list.append(parsed)

    _TENANT_SKILLS_CACHE[cache_key] = (now, parsed_list)
    return parsed_list, is_cloud_mode


def _index_all_skills(skills_root: Optional[Path] = None, include_cloud: bool = True) -> List[Dict[str, Any]]:
    """Discovers and parses metadata for all SKILL.md files on disk and active cloud ai_skill records."""
    root = skills_root or _resolve_skills_dir()
    indexed_by_slug: Dict[str, Dict[str, Any]] = {}

    if root.is_dir():
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
                    content, meta, body = "", {}, ""

                title = meta.get("name") or slug
                desc = meta.get("description") or ""

                tags = []
                if isinstance(meta.get("tags"), list):
                    tags = meta.get("tags", [])
                elif isinstance(meta.get("metadata"), dict) and isinstance(meta["metadata"].get("hermes"), dict):
                    tags = meta["metadata"]["hermes"].get("tags", [])

                indexed_by_slug[slug] = {
                    "id": meta.get("id"),
                    "category": category,
                    "slug": slug,
                    "name": title,
                    "description": desc,
                    "body": content,
                    "body_preview": body[:500] if body else "",
                    "version": meta.get("version", "1.0.0"),
                    "author": meta.get("author", "Valstorm"),
                    "tags": tags,
                    "ownership": meta.get("ownership", "bundled"),
                    "source": "disk",
                    "path": str(skill_file),
                    "directory": str(skill_file.parent),
                    "size_bytes": skill_file.stat().st_size,
                }

    if include_cloud:
        cloud_skills, is_cloud_mode = _get_cached_cloud_skills()
        for c_skill in cloud_skills:
            c_slug = c_skill["slug"]
            if c_slug not in indexed_by_slug:
                indexed_by_slug[c_slug] = dict(c_skill)
            elif is_cloud_mode:
                # In Cloud mode, tenant DB record overrides static container file while keeping fallback directory for subfiles
                existing_disk = indexed_by_slug[c_slug]
                merged = dict(c_skill)
                if not merged.get("directory") and existing_disk.get("directory"):
                    merged["directory"] = existing_disk["directory"]
                indexed_by_slug[c_slug] = merged
            else:
                # In Host mode, enrich disk skill with cloud record ID if missing
                if c_skill.get("id") and not indexed_by_slug[c_slug].get("id"):
                    indexed_by_slug[c_slug]["id"] = c_skill["id"]

    return sorted(list(indexed_by_slug.values()), key=lambda x: (x["category"], x["slug"]))


def get_skill_id_to_slug_map(skills_root: Optional[Path] = None) -> Dict[str, str]:
    """Builds a mapping of `aisk_...` record IDs to canonical skill slugs across cloud and disk."""
    mapping: Dict[str, str] = {}
    for s in _index_all_skills(skills_root=skills_root, include_cloud=True):
        s_id = s.get("id")
        s_slug = s.get("slug")
        if s_id and s_slug:
            mapping[str(s_id).strip()] = str(s_slug).strip()
    return mapping


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
    skills = await asyncio.to_thread(_index_all_skills, root)
    if not skills:
        if not root.is_dir():
            return f"No skills directory found at '{root}'."
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
    skills = await asyncio.to_thread(_index_all_skills, root)
    if not skills and not root.is_dir():
        return f"Error: Skills directory not found at '{root}'."

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

    # If subfile requested
    if file_path and file_path.strip():
        skill_dir_str = matched_skill.get("directory") or ""
        if not skill_dir_str:
            return f"Error: Subfile '{file_path}' not found in cloud skill '{matched_skill['slug']}'."
        sub_p = (Path(skill_dir_str) / file_path.strip()).resolve()
        # Security boundary check
        if not str(sub_p).startswith(str(Path(skill_dir_str).resolve())):
            return f"Error: Access denied: '{file_path}' traverses outside the skill directory."
        if not sub_p.is_file():
            return f"Error: Subfile '{file_path}' not found in skill directory '{matched_skill['slug']}'."
        try:
            content = await asyncio.to_thread(sub_p.read_text, encoding="utf-8", errors="replace")
            header = f"=== SKILL: {matched_skill['category']}/{matched_skill['slug']} (v{matched_skill['version']}) ===\n"
            return _truncate_output(header + content)
        except Exception as e:
            return f"Error reading skill file '{sub_p}': {e}"

    if matched_skill.get("source") == "cloud" or not matched_skill.get("path"):
        content = str(matched_skill.get("body") or "")
        header = f"=== SKILL: {matched_skill['category']}/{matched_skill['slug']} (v{matched_skill['version']}) ===\n"
        return _truncate_output(header + content)

    target_file = Path(matched_skill["path"])
    try:
        content = await asyncio.to_thread(target_file.read_text, encoding="utf-8", errors="replace")
        header = f"=== SKILL: {matched_skill['category']}/{matched_skill['slug']} (v{matched_skill['version']}) ===\n"
        return _truncate_output(header + content)
    except Exception as e:
        return f"Error reading skill file '{target_file}': {e}"


def _bump_semver_patch(version_str: str) -> str:
    """Increments the patch segment of a semantic version string (e.g. 1.0.0 -> 1.0.1)."""
    clean = (version_str or "1.0.0").strip().lstrip("v")
    parts = clean.split(".")
    if len(parts) == 3 and parts[2].isdigit():
        return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
    return "1.0.1"


def _compile_skill_markdown(
    *,
    name: str,
    slug: str,
    category: str,
    description: str,
    version: str,
    ownership: str,
    tags: List[str],
    body_content: str,
    author: str = "Valstorm Agent",
) -> str:
    """Compiles a canonical SKILL.md document with YAML frontmatter and markdown body."""
    _, stripped_body = _parse_frontmatter(body_content or "")
    final_body = (stripped_body or body_content or f"# {name}\n").strip()
    safe_desc = (description or "").replace('"', '\\"').replace("\n", " ").strip()
    tags_list_str = ", ".join(str(t).strip() for t in (tags or []) if str(t).strip())

    frontmatter_lines = [
        "---",
        f"name: {name}",
        f"slug: {slug}",
        f"category: {category}",
        f'description: "{safe_desc}"',
        f"version: {version}",
        f"author: {author}",
        f"ownership: {ownership}",
    ]
    if tags_list_str:
        frontmatter_lines.extend([
            "metadata:",
            "  hermes:",
            f"    tags: [{tags_list_str}]",
        ])
    frontmatter_lines.append("---")

    return "\n".join(frontmatter_lines) + "\n\n" + final_body + "\n"


def _append_to_markdown_section(markdown_body: str, section_name: str, content_to_append: str) -> str:
    """Appends content under an existing `## <section_name>` heading or creates the section at the end."""
    clean_sec = section_name.strip().lstrip("#").strip()
    clean_item = content_to_append.strip()
    if not clean_item:
        return markdown_body

    lines = markdown_body.splitlines()
    sec_pattern = re.compile(rf"^##\s+{re.escape(clean_sec)}\s*$", re.IGNORECASE)

    sec_start_idx = None
    for idx, line in enumerate(lines):
        if sec_pattern.match(line.strip()):
            sec_start_idx = idx
            break

    if sec_start_idx is None:
        base = markdown_body.rstrip()
        return f"{base}\n\n## {clean_sec}\n{clean_item}\n"

    # Find the next level-2 heading (`## `) after sec_start_idx
    insert_idx = len(lines)
    for idx in range(sec_start_idx + 1, len(lines)):
        if lines[idx].strip().startswith("## "):
            insert_idx = idx
            break

    # Trim trailing blank lines right before the next heading so items group cleanly
    while insert_idx > sec_start_idx + 1 and not lines[insert_idx - 1].strip():
        insert_idx -= 1

    updated_lines = lines[:insert_idx] + [clean_item] + lines[insert_idx:]
    return "\n".join(updated_lines).rstrip() + "\n"


async def _persist_skill_to_cloud(
    record_payload: Dict[str, Any],
    existing_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Creates or updates an `ai_skill` record in the authenticated tenant's MongoDB via ValstormApiClient."""
    try:
        from tools.valstorm_client import ValstormApiClient, get_current_valstorm_auth
        scoped_auth = get_current_valstorm_auth()
    except Exception:
        scoped_auth = None

    if not scoped_auth or not scoped_auth[0] or not scoped_auth[1]:
        return None

    token, base_url = scoped_auth
    try:
        async with ValstormApiClient(token=token, base_url=base_url) as client:
            if existing_id:
                upd_payload = dict(record_payload)
                upd_payload["id"] = existing_id
                res = await client.records_update("ai_skill", [upd_payload])
            else:
                res = await client.records_create("ai_skill", [record_payload])
            if isinstance(res, list) and res:
                return res[0] if isinstance(res[0], dict) else {"id": existing_id}
            if isinstance(res, dict):
                recs = res.get("records") or res.get("data")
                if isinstance(recs, list) and recs and isinstance(recs[0], dict):
                    return recs[0]
                return res
    except Exception:
        pass
    return None


async def _delete_skill_from_cloud(record_id: str) -> bool:
    """Deletes an `ai_skill` record from the authenticated tenant's MongoDB."""
    if not record_id:
        return False
    try:
        from tools.valstorm_client import ValstormApiClient, get_current_valstorm_auth
        scoped_auth = get_current_valstorm_auth()
    except Exception:
        scoped_auth = None

    if not scoped_auth or not scoped_auth[0] or not scoped_auth[1]:
        return False

    token, base_url = scoped_auth
    try:
        async with ValstormApiClient(token=token, base_url=base_url) as client:
            await client.records_delete("ai_skill", [record_id])
            return True
    except Exception:
        return False


def _resolve_writable_skills_dir() -> Path:
    """Resolves the writable root directory for storing created or adopted skills on disk."""
    custom_env = os.environ.get("VALSTORM_SKILLS_DIR")
    if custom_env and custom_env.strip():
        p = Path(custom_env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    home_skills = (Path.home() / ".valstorm" / "skills").resolve()
    home_skills.mkdir(parents=True, exist_ok=True)
    return home_skills


async def execute_skill_mutation(
    action: str,
    slug: str,
    category: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    body: Optional[str] = None,
    section: Optional[str] = None,
    content_to_append: Optional[str] = None,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Executes a validated skill mutation (create, patch, adopt, delete) and returns a structured receipt."""
    act = (action or "").strip().lower()
    if act not in ("create", "patch", "adopt", "delete"):
        return {
            "status": "error",
            "message": f"Error: Invalid action '{action}'. Must be one of: 'create', 'patch', 'adopt', 'delete'.",
        }

    raw_slug = (slug or "").strip().lower()
    if "/" in raw_slug and not category:
        category, raw_slug = raw_slug.split("/", 1)
    clean_slug = re.sub(r"[^a-z0-9\-]", "-", raw_slug.replace("_", "-").strip("-"))
    if not clean_slug or len(clean_slug) < 2:
        return {"status": "error", "message": "Error: A valid kebab-case skill `slug` is required."}

    try:
        from tools.valstorm_client import get_current_tenant_context
        ctx = get_current_tenant_context()
    except Exception:
        ctx = None

    is_cloud_mode = bool(ctx and ctx.is_cloud_mode)
    org_id = ctx.org_id if ctx else None

    skills = await asyncio.to_thread(_index_all_skills)
    existing = next((s for s in skills if s["slug"] == clean_slug), None)

    # 1. Handle DELETE
    if act == "delete":
        if not existing:
            return {"status": "error", "message": f"Error: Skill '{clean_slug}' not found."}
        if existing.get("ownership") == "bundled":
            return {
                "status": "error",
                "message": (
                    f"Error: Skill '{clean_slug}' is a bundled system skill and cannot be deleted. "
                    f"Use `skill_manage(action='adopt', slug='{clean_slug}')` to customize a tenant copy."
                ),
            }
        deleted_disk = False
        if existing.get("path"):
            p = Path(existing["path"])
            if p.is_file():
                p.unlink()
                deleted_disk = True
        deleted_cloud = False
        if existing.get("id"):
            deleted_cloud = await _delete_skill_from_cloud(str(existing["id"]))

        invalidate_tenant_skills_cache(org_id)
        receipt = {
            "status": "success",
            "action": "delete",
            "slug": clean_slug,
            "deleted_disk": deleted_disk,
            "deleted_cloud": deleted_cloud,
            "message": f"Deleted skill '{clean_slug}'.",
        }
        _emit_skill_sse_event(receipt)
        return receipt

    # 2. Handle ADOPT (clone bundled skill into user/tenant ownership)
    if act == "adopt":
        if not existing:
            return {"status": "error", "message": f"Error: Skill '{clean_slug}' not found to adopt."}
        new_ownership = "tenant" if is_cloud_mode else "user"
        raw_existing_content = existing.get("body") or ""
        if not raw_existing_content and existing.get("path") and Path(existing["path"]).is_file():
            raw_existing_content = Path(existing["path"]).read_text(encoding="utf-8", errors="replace")
        _, parsed_body = _parse_frontmatter(raw_existing_content)

        compiled = _compile_skill_markdown(
            name=str(existing.get("name") or clean_slug),
            slug=clean_slug,
            category=str(existing.get("category") or category or "general"),
            description=str(existing.get("description") or ""),
            version=str(existing.get("version") or "1.0.0"),
            ownership=new_ownership,
            tags=list(existing.get("tags") or []),
            body_content=parsed_body,
            author=str(existing.get("author") or "Valstorm"),
        )
        target_path_str = existing.get("path") or ""
        if not is_cloud_mode:
            target_file = (
                Path(target_path_str)
                if target_path_str
                else (_resolve_writable_skills_dir() / str(existing.get("category") or "general") / clean_slug / "SKILL.md")
            )
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(compiled, encoding="utf-8")
            target_path_str = str(target_file)

        cloud_rec = await _persist_skill_to_cloud(
            {
                "name": str(existing.get("name") or clean_slug),
                "api_name": clean_slug,
                "category": str(existing.get("category") or category or "general"),
                "description": str(existing.get("description") or ""),
                "body": compiled,
                "is_active": True,
            },
            existing_id=str(existing["id"]) if existing.get("id") else None,
        )
        invalidate_tenant_skills_cache(org_id)
        receipt = {
            "status": "success",
            "action": "adopt",
            "slug": clean_slug,
            "category": str(existing.get("category") or "general"),
            "version": str(existing.get("version") or "1.0.0"),
            "ownership": new_ownership,
            "path": target_path_str,
            "record_id": cloud_rec.get("id") if isinstance(cloud_rec, dict) else existing.get("id"),
            "message": f"Adopted skill '{clean_slug}' into '{new_ownership}' ownership. It can now be freely patched.",
        }
        _emit_skill_sse_event(receipt)
        return receipt

    # 3. Handle CREATE
    if act == "create":
        eff_cat = re.sub(r"[^a-z0-9\-]", "-", (category or (existing.get("category") if existing else None) or "general").strip().lower())
        eff_name = (name or (existing.get("name") if existing else None) or clean_slug.replace("-", " ").title()).strip()
        eff_desc = (description or (existing.get("description") if existing else None) or f"Procedural guide for {eff_name}.").strip()
        eff_tags = tags if tags is not None else (list(existing.get("tags") or []) if existing else [])
        eff_version = "1.0.0"
        eff_ownership = "tenant" if is_cloud_mode else "user"
        eff_body = (body or f"# {eff_name}\n\n## Overview\n{eff_desc}\n").strip()

        compiled = _compile_skill_markdown(
            name=eff_name,
            slug=clean_slug,
            category=eff_cat,
            description=eff_desc,
            version=eff_version,
            ownership=eff_ownership,
            tags=eff_tags,
            body_content=eff_body,
        )

        from core.security_guard import SecurityGuard
        is_safe, sec_err = SecurityGuard.validate_file_content(f"{clean_slug}/SKILL.md", compiled)
        if not is_safe:
            return {"status": "error", "message": f"Error: Security validation failed for skill '{clean_slug}': {sec_err}"}

        target_path_str = ""
        if not is_cloud_mode:
            writable_root = _resolve_writable_skills_dir()
            target_file = writable_root / eff_cat / clean_slug / "SKILL.md"
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(compiled, encoding="utf-8")
            target_path_str = str(target_file)

        cloud_rec = await _persist_skill_to_cloud(
            {
                "name": eff_name,
                "api_name": clean_slug,
                "category": eff_cat,
                "description": eff_desc,
                "body": compiled,
                "is_active": True,
            },
            existing_id=str(existing["id"]) if (existing and existing.get("id")) else None,
        )
        invalidate_tenant_skills_cache(org_id)
        rec_id = cloud_rec.get("id") if isinstance(cloud_rec, dict) else None
        receipt = {
            "status": "success",
            "action": "create",
            "slug": clean_slug,
            "category": eff_cat,
            "version": eff_version,
            "ownership": eff_ownership,
            "path": target_path_str,
            "record_id": rec_id,
            "message": f"Created skill '{eff_cat}/{clean_slug}' (v{eff_version}).",
        }
        _emit_skill_sse_event(receipt)
        return receipt

    # 4. Handle PATCH
    if not existing:
        return {
            "status": "error",
            "message": f"Error: Skill '{clean_slug}' not found to patch. Use `action='create'` to create a new skill.",
        }

    if existing.get("ownership") == "bundled":
        return {
            "status": "error",
            "message": (
                f"Error: Skill '{clean_slug}' has bundled ownership and cannot be directly patched. "
                f"Call `skill_manage(action='adopt', slug='{clean_slug}')` first to adopt a customizable copy."
            ),
        }

    raw_existing = existing.get("body") or ""
    if not raw_existing and existing.get("path") and Path(existing["path"]).is_file():
        raw_existing = Path(existing["path"]).read_text(encoding="utf-8", errors="replace")

    meta, current_body = _parse_frontmatter(raw_existing)
    updated_body = current_body or raw_existing

    if body and body.strip():
        _, updated_body = _parse_frontmatter(body.strip())
    elif section and content_to_append:
        updated_body = _append_to_markdown_section(updated_body, section, content_to_append)
    elif old_text is not None and new_text is not None:
        if old_text not in updated_body:
            return {
                "status": "error",
                "message": f"Error: `old_text` not found in skill '{clean_slug}'. Inspect with `skill_view('{clean_slug}')` first.",
            }
        updated_body = updated_body.replace(old_text, new_text, 1)
    else:
        return {
            "status": "error",
            "message": "Error: `action='patch'` requires either (`section` + `content_to_append`), (`old_text` + `new_text`), or a replacement `body`.",
        }

    eff_cat = str(category or existing.get("category") or "general").strip()
    eff_name = str(name or existing.get("name") or clean_slug).strip()
    eff_desc = str(description or existing.get("description") or "").strip()
    eff_tags = tags if tags is not None else list(existing.get("tags") or [])
    new_version = _bump_semver_patch(str(existing.get("version") or meta.get("version") or "1.0.0"))
    eff_ownership = str(existing.get("ownership") or ("tenant" if is_cloud_mode else "user"))

    compiled = _compile_skill_markdown(
        name=eff_name,
        slug=clean_slug,
        category=eff_cat,
        description=eff_desc,
        version=new_version,
        ownership=eff_ownership,
        tags=eff_tags,
        body_content=updated_body,
        author=str(existing.get("author") or "Valstorm Agent"),
    )

    from core.security_guard import SecurityGuard
    is_safe, sec_err = SecurityGuard.validate_file_content(f"{clean_slug}/SKILL.md", compiled)
    if not is_safe:
        return {"status": "error", "message": f"Error: Security validation failed for skill '{clean_slug}': {sec_err}"}

    target_path_str = existing.get("path") or ""
    if not is_cloud_mode:
        target_file = (
            Path(target_path_str)
            if target_path_str
            else (_resolve_writable_skills_dir() / eff_cat / clean_slug / "SKILL.md")
        )
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(compiled, encoding="utf-8")
        target_path_str = str(target_file)

    cloud_rec = await _persist_skill_to_cloud(
        {
            "name": eff_name,
            "api_name": clean_slug,
            "category": eff_cat,
            "description": eff_desc,
            "body": compiled,
            "is_active": True,
        },
        existing_id=str(existing["id"]) if existing.get("id") else None,
    )
    invalidate_tenant_skills_cache(org_id)
    receipt = {
        "status": "success",
        "action": "patch",
        "slug": clean_slug,
        "category": eff_cat,
        "version": new_version,
        "ownership": eff_ownership,
        "path": target_path_str,
        "record_id": cloud_rec.get("id") if isinstance(cloud_rec, dict) else existing.get("id"),
        "message": f"Patched skill '{eff_cat}/{clean_slug}' -> v{new_version}.",
    }
    _emit_skill_sse_event(receipt)
    return receipt


def _emit_skill_sse_event(receipt: Dict[str, Any]) -> None:
    """Pushes a non-blocking `skill.updated` SSE event onto the active run queue if present."""
    if receipt.get("status") != "success":
        return
    try:
        from server import get_current_event_queue
        eq = get_current_event_queue()
        if eq is not None:
            eq.put_nowait({
                "event": "skill.updated",
                "action": receipt.get("action"),
                "slug": receipt.get("slug"),
                "category": receipt.get("category"),
                "version": receipt.get("version"),
                "ownership": receipt.get("ownership"),
            })
    except Exception:
        pass


@tool
async def skill_manage(
    action: str,
    slug: str,
    category: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    body: Optional[str] = None,
    section: Optional[str] = None,
    content_to_append: Optional[str] = None,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Creates, patches, adopts, or deletes a procedural skill document (SKILL.md / ai_skill).

    Actions:
      - 'create': Creates a new reusable skill with YAML frontmatter and Markdown body.
      - 'patch': Updates an existing skill by appending to a section (`section='Pitfalls'`, `content_to_append='...'`)
                 or replacing exact text (`old_text='...'`, `new_text='...'`), automatically bumping the patch version.
      - 'adopt': Clones a read-only `bundled` skill into a customizable user/tenant copy so it can be patched.
      - 'delete': Deletes a custom user/tenant skill (bundled skills cannot be deleted).

    Args:
        action: Mutation action ('create', 'patch', 'adopt', 'delete').
        slug: Kebab-case skill identifier (e.g. 'stripe-webhook-audit' or 'valstorm-pytest-patterns').
        category: Skill category folder (e.g. 'valstorm-internal', 'software-development', 'finance').
        name: Human-readable title of the skill.
        description: Concise 1-2 line trigger description explaining when to invoke this skill.
        body: Full Markdown body when creating or replacing a skill.
        section: Section heading name (e.g. 'Pitfalls', 'Examples') when appending via action='patch'.
        content_to_append: Markdown bullet or block to append under `section` when action='patch'.
        old_text: Exact existing text to replace when action='patch'.
        new_text: Replacement text when action='patch' with `old_text`.
        tags: Optional list of discovery keywords/tags.

    Returns:
        Confirmation message with the skill slug, category, and updated semantic version.
    """
    receipt = await execute_skill_mutation(
        action=action,
        slug=slug,
        category=category,
        name=name,
        description=description,
        body=body,
        section=section,
        content_to_append=content_to_append,
        old_text=old_text,
        new_text=new_text,
        tags=tags,
    )
    return str(receipt.get("message") or json.dumps(receipt))


def create_skill_tools() -> List[Any]:
    """Returns list of skill inspection, discovery, and authoring tools."""
    return [skill_view, skill_list, skill_manage]


def register_skill_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers skill discovery, inspection, and authoring tools into the provided ToolRegistry."""
    for tool_fn in create_skill_tools():
        registry.register(tool_fn)
    return registry
