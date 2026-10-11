#!/usr/bin/env python3
"""Hydrates local Markdown catalog for Valstorm AI Skills and Agent Profiles.

Creates clean, human-and-agent-editable Markdown files with structured YAML frontmatter:
- skills/public/{category}/{slug}.md
- skills/internal/{category}/{slug}.md
- agents/public/{slug}.md
- agents/internal/{slug}.md

Enforces public vs. internal visibility segregation.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None


# Default Valstorm Org ID for proprietary internal skills
VALSTORM_INTERNAL_ORG_ID = "org_dSnPMRjS1ZkkdYQ2"

# Explicit slugs to exclude/delete
DELETED_SKILL_SLUGS: Set[str] = {
    "airtable", "apple-notes", "apple-reminders", "architecture-diagram", "ascii-art", "ascii-video",
    "baoyu-infographic", "claude-design", "codebase-memory-mcp", "comfyui", "debugging-hermes-tui-commands",
    "design-md", "document-to-action-items", "docx", "dspy", "email-inbox-triage", "evaluating-llms-harness",
    "excalidraw", "findmy", "gif-search", "google-workspace", "hermes-agent", "hermes-agent-skill-authoring",
    "hermes-api-integration", "hermes-desktop-plugins", "hermes-profile-distribution", "hermes-profile-sync",
    "himalaya", "huggingface-hub", "humanizer", "ideation", "imessage", "inspecting-hermes-desktop-dom",
    "jupyter-live-kernel", "llama-cpp", "llm-runtime-patterns", "macos-computer-use", "manim-video", "maps",
    "meeting-action-items", "nano-pdf", "notion", "obsidian", "ocr-and-documents", "openhue", "p5js", "pdf",
    "plan", "popular-web-designs", "powerpoint", "pretext", "product-price-monitor", "python-debugpy",
    "requesting-code-review", "sdlc-review", "serving-llms-vllm", "session-librarian", "simplify-code",
    "sketch", "songsee", "songwriting-and-ai-music", "spike", "spotify", "systematic-debugging",
    "teams-meeting-pipeline", "touchdesigner-mcp", "weekly-review-planning", "weights-and-biases", "xlsx",
    "xurl", "youtube-content",
}

# Acronym & special capitalization dictionary for clean title formatting
SPECIAL_WORDS = {
    "aeo": "AEO",
    "ai": "AI",
    "api": "API",
    "apis": "APIs",
    "b2b": "B2B",
    "cli": "CLI",
    "crud": "CRUD",
    "cud": "CUD",
    "dom": "DOM",
    "dspy": "DSPy",
    "fcp": "FCP",
    "fcpxml": "FCPXML",
    "html": "HTML",
    "icp": "ICP",
    "ipc": "IPC",
    "json": "JSON",
    "jwt": "JWT",
    "llm": "LLM",
    "llms": "LLMs",
    "mcp": "MCP",
    "mdx": "MDX",
    "mrr": "MRR",
    "p5js": "p5.js",
    "pdf": "PDF",
    "pnl": "PnL",
    "qa": "QA",
    "rag": "RAG",
    "rest": "REST",
    "saas": "SaaS",
    "sdlc": "SDLC",
    "seo": "SEO",
    "sql": "SQL",
    "sse": "SSE",
    "tdd": "TDD",
    "tui": "TUI",
    "uat": "UAT",
    "ui": "UI",
    "url": "URL",
    "ux": "UX",
    "vfs": "VFS",
    "vite": "Vite",
    "vllm": "vLLM",
    "xlsx": "XLSX",
}

# Active skill defaults for standard profiles
PROFILE_SKILL_DEFAULTS: Dict[str, List[str]] = {
    "developer": [
        "bash-scripting",
        "code-modification-fallbacks",
        "codebase-inspection",
        "test-driven-development",
        "valstorm-cli",
        "valstorm-backend-patterns",
        "valstorm-react-patterns",
        "node-inspect-debugger",
    ],
    "researcher": [
        "arxiv",
        "grounded-citations",
        "llm-wiki",
        "competitor-news-monitor",
        "blogwatcher",
        "valstorm-vfs",
        "vfs-search-and-rag-pipeline",
    ],
    "backend-tester": [
        "python-testing",
        "valstorm-pytest-patterns",
        "test-driven-development",
    ],
    "frontend-tester": [
        "playwright-automation",
        "playwright-electron-testing",
        "valstorm-playwright-patterns",
    ],
    "docs-writer": [
        "technical-documentation",
        "research-paper-writing",
        "valstorm-mdx-editor",
    ],
    "writer": [
        "technical-documentation",
        "research-paper-writing",
        "valstorm-mdx-editor",
    ],
    "executive-advisory": [
        "answer-engine-optimization-audit",
        "competitor-news-monitor",
        "grounded-citations",
        "saas-metrics-and-reporting",
        "saas-financial-metrics",
    ],
    "client-onboarding": [
        "client-torch-pass-protocol",
        "user-interaction-style",
    ],
}

PROFILE_TIER_DEFAULTS: Dict[str, str] = {
    "architect": "tier_1",
    "executive-advisory": "tier_1",
    "video-editor": "tier_3",
    "archivist": "tier_3",
}

PROFILE_MODEL_DEFAULTS: Dict[str, str] = {
    "tier_1": "gemini-pro-latest",
    "tier_2": "gemini-flash-latest",
    "tier_3": "gemini-flash-lite-latest",
}

INTERNAL_SKILL_CATEGORIES: Set[str] = {
    "valstorm-internal",
    "valstorm",
    "backend",
}

INTERNAL_AGENTS: Set[str] = {
    "orchestrator",
    "executive-advisory",
    "client-onboarding",
    "dev-ops",
}


def format_title_name(raw_name: Optional[str], slug: str) -> str:
    """Formats a slug or raw string into clean Title Case with proper acronym handling."""
    if raw_name and raw_name != slug and not (raw_name.islower() and ("-" in raw_name or "_" in raw_name)):
        return raw_name.strip()

    words = re.split(r"[-_ ]+", slug)
    titled = []
    for w in words:
        low = w.lower()
        if low in SPECIAL_WORDS:
            titled.append(SPECIAL_WORDS[low])
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def dump_yaml_frontmatter(metadata: Dict[str, Any]) -> str:
    """Serializes metadata dict to YAML frontmatter with consistent formatting."""
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


def parse_markdown_document(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Extracts frontmatter dict and body text from markdown string."""
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


def generate_skill_tags(slug: str, category: str, name: str) -> List[str]:
    """Generates a clean list of search tags for a skill."""
    tags_set = {category}
    for word in re.split(r"[-_ ]+", slug):
        w = word.lower().strip()
        if len(w) > 2 and w not in {"the", "and", "for", "with", "valstorm"}:
            tags_set.add(w)
    for word in re.split(r"[-_ ]+", name):
        w = word.lower().strip()
        if len(w) > 2 and w not in {"the", "and", "for", "with", "valstorm"}:
            tags_set.add(w)
    return sorted(list(tags_set))


def hydrate_catalog(
    monorepo_root: Optional[Path] = None,
    skills_source: Optional[Path] = None,
    agents_source: Optional[Path] = None,
) -> Tuple[int, int]:
    """Hydrates Markdown files in skills/ and agents/ segregated by visibility."""
    if monorepo_root is None:
        monorepo_root = Path(__file__).resolve().parent.parent.parent.parent

    skills_dir = monorepo_root / "skills"
    agents_dir = monorepo_root / "agents"
    runtime_exports = monorepo_root / "apps" / "agent-runtime" / "exports"
    valstorm_profiles_dir = Path.home() / ".valstorm" / "profiles"

    skills_public_dir = skills_dir / "public"
    skills_internal_dir = skills_dir / "internal"
    agents_public_dir = agents_dir / "public"
    agents_internal_dir = agents_dir / "internal"

    for d in [skills_public_dir, skills_internal_dir, agents_public_dir, agents_internal_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. LOAD SOURCE SKILLS
    skills_file = skills_source or runtime_exports / "ai_skills_migrated.json"
    if not skills_file.is_file():
        skills_file = monorepo_root / "ai_skills.json"
    if not skills_file.is_file():
        raise FileNotFoundError(f"Cannot find source skills JSON at {skills_file}")

    raw_skills: List[Dict[str, Any]] = json.loads(skills_file.read_text(encoding="utf-8"))

    # 2. LOAD SOURCE AGENTS
    agents_file = agents_source or runtime_exports / "ai_agents_migrated.json"
    if not agents_file.is_file():
        agents_file = monorepo_root / "ai_agents.json"
    if not agents_file.is_file():
        raise FileNotFoundError(f"Cannot find source agents JSON at {agents_file}")

    raw_agents: List[Dict[str, Any]] = json.loads(agents_file.read_text(encoding="utf-8"))

    # =========================================================================
    # PROCESS SKILLS
    # =========================================================================
    written_skills_count = 0
    all_skill_slugs: Set[str] = set()

    for s in raw_skills:
        slug = str(s.get("api_name") or s.get("slug") or s.get("name", "")).lower().strip().replace(" ", "-").replace("_", "-")
        if not slug or slug in DELETED_SKILL_SLUGS:
            continue
        all_skill_slugs.add(slug)

        category = str(s.get("category", "general")).strip()
        raw_name = format_title_name(s.get("name"), slug)
        api_name = s.get("api_name") or slug.replace("-", "_")

        # Determine visibility segregation
        if slug in ("event-telemetry-persistence", "mongodb-serialization-boundaries", "electron-development"):
            visibility = "internal"
            if slug == "electron-development":
                category = "software-development"
        elif slug in ("client-torch-pass-protocol", "answer-engine-optimization-audit"):
            visibility = "public"
        elif category in INTERNAL_SKILL_CATEGORIES or category.startswith("valstorm-"):
            visibility = "internal"
        else:
            visibility = "public"

        # Determine target directory
        target_root = skills_internal_dir if visibility in ("internal", "private") else skills_public_dir
        cat_dir = target_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_file = cat_dir / f"{slug}.md"

        body = s.get("body") or s.get("instructions") or ""
        if body.strip().startswith("---"):
            _, body = parse_markdown_document(body)

        # Apply vsagent text updates if autonomous AI agent skill
        if slug in ("claude-code", "codex", "computer-use", "merge-reconciler", "opencode"):
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

        description = s.get("description", "").replace("Hermes", "vsagent")
        if not description:
            desc_lines = [line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")]
            description = desc_lines[0] if desc_lines else f"Procedural guide for {raw_name}."

        tags = s.get("tags") or generate_skill_tags(slug, category, raw_name)

        frontmatter_data = {
            "name": raw_name,
            "slug": slug,
            "api_name": api_name,
            "category": category,
            "description": description[:350].strip(),
            "visibility": visibility,
            "is_active": s.get("is_active", True),
            "version": s.get("version", "1.0.0"),
            "tags": tags,
        }

        content = dump_yaml_frontmatter(frontmatter_data) + "\n" + body.strip() + "\n"
        out_file.write_text(content, encoding="utf-8")
        written_skills_count += 1

    # Add exemplar Agency skills (Public visibility)
    agency_skills = [
        {
            "name": "Client Torch Pass Protocol",
            "slug": "client-torch-pass-protocol",
            "api_name": "client_torch_pass_protocol",
            "category": "agency-operations",
            "description": "Standard operating procedure for transitioning an agency client from initial build/setup through training to independent operation and ongoing advisory.",
            "visibility": "public",
            "is_active": True,
            "version": "1.0.0",
            "tags": ["agency-operations", "torch-passing", "client-onboarding", "training", "handoff"],
            "body": """# Client Torch Pass Protocol Guide

## Overview & Core Philosophy
Our agency model is defined by three clear phases:
1. **Initial Build & Setup**: High-velocity infrastructure, inbound SEO/AEO scaffolding, paid media architectures, and content pipeline creation.
2. **Comprehensive Client Training & Torch Passing**: Hands-on onboarding, workflow shadowing, video-recorded SOP walkthroughs, and operational empowerment so the client team can operate the systems independently.
3. **Ongoing Market & Trend Advisory**: Bi-weekly or monthly strategic advisory on search engine algorithm updates, Answer Engine Optimization (AEO) shifts, paid media efficiency, and competitive intelligence.

## Step-by-Step Torch Passing Procedure

### 1. Verification of Build Deliverables
- Audit all CRM schemas, tracking pixels, search analytics, and content templates.
- Confirm all permissions, API credentials, and administrative accesses are transferred to the client.

### 2. Client Training Modules
- Conduct live interactive training sessions covering:
  - Daily Inbound Pipeline Monitoring & Lead Qualification.
  - Content Publishing & AEO Best Practices.
  - Paid Ads Budget Management & Bid Optimization.
- Save all recorded session links into the client's Valstorm Cloud Knowledge Vault (`02_SOPs/Client_Training/`).

### 3. Formal Torch Passing Sign-Off
- Review the Operational Checklist with the client executive sponsor.
- Issue the official Torch Pass Certificate and Transition Summary.

### 4. Transition to Trend Advisory
- Schedule recurring executive advisory sessions.
- Establish automated alert channels for keyword volatility, AEO citation shifts, and competitor movements.
""",
        },
        {
            "name": "Answer Engine Optimization (AEO) Audit",
            "slug": "answer-engine-optimization-audit",
            "api_name": "answer_engine_optimization_audit",
            "category": "marketing-strategy",
            "description": "Audit framework and actionable playbook for optimizing brand visibility across AI answer engines (ChatGPT Search, Perplexity, Gemini, Google AI Overviews).",
            "visibility": "public",
            "is_active": True,
            "version": "1.0.0",
            "tags": ["marketing-strategy", "aeo", "seo", "answer-engines", "inbound-search", "ai-visibility"],
            "body": """# Answer Engine Optimization (AEO) Audit Guide

## Purpose & Scope
Answer Engine Optimization (AEO) is the practice of structuring digital assets, domain authority, structured data, and citations so that LLMs and AI search engines synthesize and recommend our client as the primary authoritative answer.

## Key Audit Vectors

### 1. Brand Citation Footprint
- Inspect presence on Wikidata, Crunchbase, G2, Trustpilot, and industry directories.
- Measure citation frequency across queries in Perplexity, ChatGPT Search, and Google AI Overviews.

### 2. Information Architecture & Fact Density
- Ensure high information density: clear definitions, comparative tables, exact numerical specifications, and concise direct answers in `<h2>` and paragraph summaries.
- Apply JSON-LD Schema.org markup (`Organization`, `Product`, `FAQPage`, `HowTo`).

### 3. Entity Graph & Sentiment Alignment
- Verify consensus facts regarding pricing, capabilities, and executive leadership across digital sources.
- Remediate conflicting or outdated brand data across third-party blogs and aggregator sites.

## Deliverables
1. **AEO Visibility Scorecard (0-100)**: Benchmark vs top 3 competitors.
2. **Remediation Roadmap**: High-priority structured data and content tweaks.
3. **Monthly Tracking**: Monitored query clusters across answer engines.
""",
        },
    ]

    for askill in agency_skills:
        slug = askill["slug"]
        all_skill_slugs.add(slug)
        cat = askill["category"]
        cat_dir = skills_public_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_file = cat_dir / f"{slug}.md"
        fm = {
            "name": askill["name"],
            "slug": slug,
            "api_name": askill["api_name"],
            "category": cat,
            "description": askill["description"],
            "visibility": askill["visibility"],
            "is_active": askill["is_active"],
            "version": askill["version"],
            "tags": askill["tags"],
        }
        content = dump_yaml_frontmatter(fm) + "\n" + askill["body"].strip() + "\n"
        out_file.write_text(content, encoding="utf-8")
        written_skills_count += 1

    # =========================================================================
    # PROCESS AGENTS
    # =========================================================================
    written_agents_count = 0
    processed_agent_slugs: Set[str] = set()

    for a in raw_agents:
        slug = str(a.get("api_name") or a.get("slug") or a.get("name", "")).lower().strip().replace(" ", "-").replace("_", "-")
        if not slug:
            continue
        processed_agent_slugs.add(slug)

        raw_name = format_title_name(a.get("name"), slug)
        api_name = a.get("api_name") or slug.replace("-", "_")
        desc = a.get("description", "")

        # Determine visibility
        explicit_vis = a.get("visibility")
        if explicit_vis in ("public", "internal", "private"):
            visibility = explicit_vis
        elif slug in INTERNAL_AGENTS:
            visibility = "internal"
        else:
            visibility = "public"

        model_tier = a.get("model_tier") or PROFILE_TIER_DEFAULTS.get(slug, "tier_2")
        model = a.get("model") or PROFILE_MODEL_DEFAULTS.get(model_tier, "gemini-flash-latest")
        provider = (a.get("provider") or "valstorm").lower()
        if provider in ("valstorm", "managed", "hosted"):
            provider = "valstorm"
        elif provider == "gemini":
            provider = "gemini"
        elif provider in ("claude", "anthropic"):
            provider = "anthropic"
        elif provider in ("chatgpt", "openai", "gpt"):
            provider = "openai"

        allowed_tools = a.get("allowed_tools") or [
            "execute_code",
            "terminal_exec",
            "patch_file",
            "write_file",
            "read_file",
            "search_files",
            "calculator",
            "clarify",
            "memory_manage",
            "session_search",
            "skill_view",
            "skill_list",
        ]

        # Gather skills & filter deleted
        skills_attached = a.get("ai_skills") or a.get("skills") or []
        clean_attached = [
            s for s in skills_attached
            if isinstance(s, str) and s in all_skill_slugs and s not in DELETED_SKILL_SLUGS
        ]
        if slug in PROFILE_SKILL_DEFAULTS:
            for s_default in PROFILE_SKILL_DEFAULTS[slug]:
                if s_default not in clean_attached and s_default in all_skill_slugs and s_default not in DELETED_SKILL_SLUGS:
                    clean_attached.append(s_default)

        system_prompt = a.get("system_prompt") or f"You are {raw_name}. Fulfill user goals diligently."

        frontmatter_data = {
            "name": raw_name,
            "slug": slug,
            "api_name": api_name,
            "description": desc,
            "visibility": visibility,
            "model_tier": model_tier,
            "model": model,
            "provider": provider,
            "is_active": a.get("is_active", True),
            "allowed_tools": allowed_tools,
            "skills": sorted(list(set(clean_attached))),
        }

        target_dir = agents_internal_dir if visibility in ("internal", "private") else agents_public_dir
        out_file = target_dir / f"{slug}.md"

        content = dump_yaml_frontmatter(frontmatter_data) + "\n" + system_prompt.strip() + "\n"
        out_file.write_text(content, encoding="utf-8")
        written_agents_count += 1

    # Add exemplar Agency agents
    exemplar_agents = [
        {
            "name": "Executive Marketing & Trend Advisory",
            "slug": "executive-advisory",
            "api_name": "executive_advisory",
            "description": "Provides high-level inbound search, SEO, AEO, paid ads, and market trend advisory for marketing agency leadership and enterprise clients.",
            "visibility": "internal",
            "model_tier": "tier_1",
            "model": "gemini-pro-latest",
            "provider": "valstorm",
            "is_active": True,
            "allowed_tools": [
                "valstorm_sql_query",
                "valstorm_vfs_search",
                "valstorm_vfs_browse",
                "valstorm_vfs_get_file",
                "read_file",
                "search_files",
                "execute_code",
                "calculator",
                "clarify",
                "memory_manage",
                "session_search",
                "skill_view",
                "skill_list",
            ],
            "skills": [
                "answer-engine-optimization-audit",
                "competitor-news-monitor",
                "grounded-citations",
                "saas-metrics-and-reporting",
                "saas-financial-metrics",
            ],
            "system_prompt": (
                "You are the Executive Marketing & Trend Advisory Specialist for our 40-person inbound marketing agency.\n\n"
                "## Agency Business Model Context:\n"
                "We focus on inbound search via SEO, AEO (Answer Engine Optimization), Paid Ads, and Content. "
                "Our core engagement model is an initial build/setup, followed by comprehensive client training & torch passing, "
                "and sustained ongoing trend/market advisory.\n\n"
                "## Your Role:\n"
                "1. Advise leadership and clients on search market developments, algorithm shifts, and LLM answer engine indexation changes.\n"
                "2. Formulate strategic growth recommendations across organic search, paid acquisition, and content syndication.\n"
                "3. Synthesize competitive research and citation analytics into high-impact executive briefings.\n"
                "4. Maintain a clear, authoritative, highly strategic consulting tone."
            ),
        },
        {
            "name": "Client Onboarding & Torch Passing Specialist",
            "slug": "client-onboarding",
            "api_name": "client_onboarding",
            "description": "Coordinates initial client setup, comprehensive workflow training, and the formal torch-passing protocol for inbound search and marketing operations.",
            "visibility": "internal",
            "model_tier": "tier_2",
            "model": "gemini-flash-latest",
            "provider": "valstorm",
            "is_active": True,
            "allowed_tools": [
                "valstorm_sql_query",
                "valstorm_record_cud",
                "valstorm_vfs_search",
                "valstorm_vfs_browse",
                "valstorm_vfs_get_file",
                "read_file",
                "search_files",
                "write_file",
                "patch_file",
                "calculator",
                "clarify",
                "memory_manage",
                "session_search",
                "skill_view",
                "skill_list",
            ],
            "skills": [
                "client-torch-pass-protocol",
                "user-interaction-style",
            ],
            "system_prompt": (
                "You are the Client Onboarding & Torch Passing Specialist for our marketing agency.\n\n"
                "## Agency Engagement Lifecycle:\n"
                "1. **Initial Setup**: Ensure all client tracking, schemas, knowledge vaults, and workflows are structured.\n"
                "2. **Training & Torch Pass**: Conduct comprehensive training so client teams operate independently.\n"
                "3. **Advisory Transition**: Hand off smoothly to the ongoing executive trend advisory team.\n\n"
                "## Your Responsibilities:\n"
                "1. Guide new clients through onboarding checklists and SOP walkthroughs.\n"
                "2. Verify training milestone completion using the Client Torch Pass Protocol.\n"
                "3. Populate the client's Valstorm Cloud Knowledge Vaults with personalized guides and recorded sessions."
            ),
        },
        {
            "name": "Technical Documentation Writer",
            "slug": "writer",
            "api_name": "writer",
            "description": "Specialized in technical documentation, API specifications, and architecture specs.",
            "visibility": "public",
            "model_tier": "tier_2",
            "model": "gemini-flash-latest",
            "provider": "valstorm",
            "is_active": True,
            "allowed_tools": [
                "read_file",
                "search_files",
                "write_file",
                "patch_file",
                "analyze_image",
                "vision_analyze",
                "skill_view",
                "skill_list",
            ],
            "skills": [
                "technical-documentation",
                "research-paper-writing",
                "valstorm-mdx-editor",
            ],
            "system_prompt": "You are a senior technical writer subagent. Author clear, concise, and structured documentation, user guides, and architecture specifications.",
        },
    ]

    for a_ex in exemplar_agents:
        slug = a_ex["slug"]
        if slug not in processed_agent_slugs:
            vis = a_ex["visibility"]
            target_dir = agents_internal_dir if vis in ("internal", "private") else agents_public_dir
            out_file = target_dir / f"{slug}.md"
            fm = {
                "name": a_ex["name"],
                "slug": slug,
                "api_name": a_ex["api_name"],
                "description": a_ex["description"],
                "visibility": vis,
                "model_tier": a_ex["model_tier"],
                "model": a_ex["model"],
                "provider": a_ex["provider"],
                "is_active": a_ex["is_active"],
                "allowed_tools": a_ex["allowed_tools"],
                "skills": [s for s in a_ex["skills"] if s in all_skill_slugs and s not in DELETED_SKILL_SLUGS],
            }
            content = dump_yaml_frontmatter(fm) + "\n" + a_ex["system_prompt"].strip() + "\n"
            out_file.write_text(content, encoding="utf-8")
            written_agents_count += 1
            processed_agent_slugs.add(slug)

    print("\n" + "=" * 70)
    print("✅ VALSTORM MARKDOWN CATALOG HYDRATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  • Total Skills Created: {written_skills_count}")
    print(f"    - Public Skills ({skills_public_dir}): {len(list(skills_public_dir.glob('*/*.md')))}")
    print(f"    - Internal Skills ({skills_internal_dir}): {len(list(skills_internal_dir.glob('*/*.md')))}")
    print(f"  • Total Agents Created: {written_agents_count}")
    print(f"    - Public Agents ({agents_public_dir}): {len(list(agents_public_dir.glob('*.md')))}")
    print(f"    - Internal Agents ({agents_internal_dir}): {len(list(agents_internal_dir.glob('*.md')))}")
    print("=" * 70 + "\n")

    return written_skills_count, written_agents_count


def main():
    parser = argparse.ArgumentParser(description="Hydrate Markdown skills and agents catalog with visibility segregation")
    parser.add_argument("--root", type=str, default=None, help="Monorepo root directory")
    parser.add_argument("--skills-source", type=str, default=None, help="Source JSON for skills")
    parser.add_argument("--agents-source", type=str, default=None, help="Source JSON for agents")

    args = parser.parse_args()
    root_dir = Path(args.root).resolve() if args.root else None
    skills_src = Path(args.skills_source).resolve() if args.skills_source else None
    agents_src = Path(args.agents_source).resolve() if args.agents_source else None

    hydrate_catalog(monorepo_root=root_dir, skills_source=skills_src, agents_source=agents_src)


if __name__ == "__main__":
    main()
