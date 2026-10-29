"""Declarative Fact Memory Store for Agent Runtime.

Maintains durable, high-signal facts across sessions:
- user: User profile, communication style, preferences
- memory: Environment facts, repository conventions, tool quirks, lessons learned
"""

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import httpx

DEFAULT_MEMORY_FILE = Path.home() / ".valstorm" / "memories.json"
_CRITICAL_FACT_RE = re.compile(
    r"\b(CRITICAL|GUARDRAIL|NEVER|ALWAYS|SAFETY PROTOCOL|MANDATORY)\b",
    re.IGNORECASE,
)


def _select_budgeted_facts(
    facts: List[str],
    query_hint: Optional[str] = None,
    max_facts: int = 25,
    max_chars_budget: int = 2000,
) -> List[str]:
    """Selects and orders facts within a strict count and character budget.

    Prioritizes:
      1. Critical guardrails (`CRITICAL`, `NEVER`, `ALWAYS`, `GUARDRAIL`)
      2. Query-contextual facts (token overlap with `query_hint`)
      3. Recency (most recently added/reconciled facts at the end of `facts`)
    """
    if not facts or max_facts <= 0 or max_chars_budget <= 0:
        return []

    query_tokens = _normalize_tokens(query_hint) if query_hint else set()
    total = len(facts)

    scored: List[tuple[tuple[int, int, int, float], int, str]] = []
    for idx, f in enumerate(facts):
        clean_f = f.strip()
        if not clean_f:
            continue
        is_crit = 1 if _CRITICAL_FACT_RE.search(clean_f) else 0
        overlap = len(_normalize_tokens(clean_f) & query_tokens) if query_tokens else 0
        has_overlap = 1 if overlap > 0 else 0
        recency = idx / max(total, 1)
        scored.append(((is_crit, has_overlap, overlap, recency), idx, clean_f))

    # Sort descending by priority score
    scored.sort(key=lambda item: item[0], reverse=True)

    selected_with_idx: List[tuple[int, str]] = []
    used_chars = 0
    for _score, orig_idx, text in scored:
        if len(selected_with_idx) >= max_facts:
            break
        line_len = len(text) + 4
        if used_chars + line_len > max_chars_budget and selected_with_idx:
            continue
        selected_with_idx.append((orig_idx, text))
        used_chars += line_len

    # Preserve critical rules first, then chronological order for readability
    selected_with_idx.sort(key=lambda pair: (0 if _CRITICAL_FACT_RE.search(pair[1]) else 1, pair[0]))
    return [text for _idx, text in selected_with_idx]


def _sanitize_scope_segment(segment: Optional[str], fallback: str) -> str:
    """Sanitizes org_id / user_id strings for safe filesystem path construction."""
    if not segment or not str(segment).strip():
        return fallback
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(segment).strip())
    return cleaned or fallback


def _resolve_scoped_memory_path(
    file_path: Optional[Union[str, Path]] = None,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    is_cloud_mode: Optional[bool] = None,
) -> Path:
    """Resolves memory file path with tenant isolation in Cloud mode and standard ~/.valstorm/memories.json in Host mode."""
    if file_path is not None:
        return Path(file_path).expanduser().resolve()

    try:
        from tools.valstorm_client import get_current_tenant_context
        ctx = get_current_tenant_context()
    except Exception:
        ctx = None

    eff_org = org_id or (ctx.org_id if ctx else None)
    eff_user = user_id or (ctx.user_id if ctx else None)
    if is_cloud_mode is not None:
        eff_cloud = bool(is_cloud_mode)
    elif ctx is not None:
        eff_cloud = bool(ctx.is_cloud_mode)
    else:
        eff_cloud = os.environ.get("VALSTORM_AGENT_MODE", "host").strip().lower() == "cloud"

    home_dir = Path(os.environ.get("HOME") or str(Path.home())).expanduser().resolve()
    if eff_cloud or (org_id and user_id):
        safe_org = _sanitize_scope_segment(eff_org, "org_anonymous")
        safe_user = _sanitize_scope_segment(eff_user, "user_default")
        return home_dir / ".valstorm" / "tenants" / safe_org / safe_user / "memories.json"

    return home_dir / ".valstorm" / "memories.json"


def _normalize_tokens(text: str) -> set[str]:
    """Tokenizes text into normalized lower-case word stems for similarity checks."""
    words = re.findall(r"\b[a-z0-9_]+\b", text.lower())
    stopwords = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "are", "we", "i", "our", "all"}
    return {w for w in words if w not in stopwords and len(w) > 1}


def _token_similarity(s1: str, s2: str) -> float:
    """Calculates Jaccard token overlap between two fact strings."""
    t1 = _normalize_tokens(s1)
    t2 = _normalize_tokens(s2)
    if not t1 or not t2:
        return 0.0
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return intersection / union if union > 0 else 0.0


class MemoryStore:
    """Manages persistent declarative facts for the agent."""

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_cloud_mode: Optional[bool] = None,
    ):
        self._explicit_file_path = file_path is not None
        self.org_id = org_id
        self.user_id = user_id
        self.is_cloud_mode = is_cloud_mode
        self.file_path = _resolve_scoped_memory_path(
            file_path=file_path,
            org_id=org_id,
            user_id=user_id,
            is_cloud_mode=is_cloud_mode,
        )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_store()

    def _init_store(self):
        if not self.file_path.is_file():
            initial_data = {"user": [], "memory": []}
            try:
                self.file_path.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _read(self) -> Dict[str, List[str]]:
        if not self.file_path.is_file():
            return {"user": [], "memory": []}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        "user": [str(x) for x in data.get("user", []) if str(x).strip()],
                        "memory": [str(x) for x in data.get("memory", []) if str(x).strip()],
                    }
        except Exception:
            return {"user": [], "memory": []}
        return {"user": [], "memory": []}

    def _write(self, data: Dict[str, List[str]]):
        temp = self.file_path.with_suffix(".tmp_mem")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp.replace(self.file_path)

    def get_facts(self, target: str = "all") -> Dict[str, List[str]]:
        """Returns facts for 'user', 'memory', or both ('all')."""
        data = self._read()
        target_clean = target.lower().strip()
        if target_clean in ("user", "profile"):
            return {"user": data.get("user", [])}
        elif target_clean in ("memory", "notes"):
            return {"memory": data.get("memory", [])}
        return data

    def add_fact(self, target: str, content: str) -> str:
        """Adds a fact to the specified memory category if not already present."""
        cat = "user" if target.lower().strip() in ("user", "profile") else "memory"
        clean_content = content.strip()
        if not clean_content:
            return "Error: Fact content cannot be empty."

        data = self._read()
        if clean_content in data[cat]:
            return f"Fact already exists in [{cat}] store: '{clean_content}'"

        data[cat].append(clean_content)
        self._write(data)
        return f"Successfully added fact to [{cat}] store: '{clean_content}'"

    def remove_fact(self, target: str, old_text: str) -> bool:
        """Removes a fact containing old_text from the specified memory category."""
        cat = "user" if target.lower().strip() in ("user", "profile") else "memory"
        clean_search = old_text.strip().lower()
        if not clean_search:
            return False

        data = self._read()
        initial_len = len(data[cat])
        data[cat] = [
            fact for fact in data[cat]
            if clean_search not in fact.lower()
        ]

        if len(data[cat]) < initial_len:
            self._write(data)
            return True
        return False

    def reconcile_fact(
        self,
        target: str,
        content: str,
        supersedes: Optional[str] = None,
        similarity_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Reconciles a new fact: deduplicates near-identical items, removes superseded facts, and appends."""
        cat = "user" if target.lower().strip() in ("user", "profile") else "memory"
        clean_content = content.strip()
        if not clean_content:
            return {"status": "error", "message": "Fact content cannot be empty."}

        data = self._read()
        existing_facts = data.get(cat, [])
        removed_facts = []

        # 1. Exact match check
        if clean_content in existing_facts:
            return {
                "status": "noop",
                "action": "exact_duplicate",
                "target": cat,
                "fact": clean_content,
            }

        # 2. Check explicit supersession or keyword conflict
        if supersedes and supersedes.strip():
            sup_clean = supersedes.strip().lower()
            retained = []
            for f in existing_facts:
                if sup_clean in f.lower():
                    removed_facts.append(f)
                else:
                    retained.append(f)
            existing_facts = retained

        # 3. Fuzzy similarity & Near-duplicate replacement
        retained = []
        for f in existing_facts:
            sim = _token_similarity(clean_content, f)
            if sim >= similarity_threshold:
                # Near identical: replace older wording with the fresh wording
                removed_facts.append(f)
            else:
                retained.append(f)

        retained.append(clean_content)
        data[cat] = retained
        self._write(data)

        action = "superseded" if removed_facts else "added"
        return {
            "status": "success",
            "action": action,
            "target": cat,
            "fact": clean_content,
            "removed_facts": removed_facts,
        }

    async def reconcile_fact_async(
        self,
        target: str,
        content: str,
        supersedes: Optional[str] = None,
        source_chat_id: Optional[str] = None,
        confidence: float = 1.0,
        similarity_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Reconciles a fact locally and synchronizes it to the tenant's cloud working memory when authenticated."""
        local_res = self.reconcile_fact(
            target=target,
            content=content,
            supersedes=supersedes,
            similarity_threshold=similarity_threshold,
        )
        if local_res.get("status") == "error":
            return local_res

        try:
            from tools.valstorm_client import get_current_valstorm_auth
            scoped_auth = get_current_valstorm_auth()
        except Exception:
            scoped_auth = None

        if scoped_auth and scoped_auth[0] and scoped_auth[1]:
            token, base_url = scoped_auth
            cat = "user" if target.lower().strip() in ("user", "profile") else "memory"
            cloud_category = "user_fact" if cat == "user" else "environment_note"
            clean_base = base_url.rstrip("/")
            if not clean_base.endswith("/v1") and "/v1" not in clean_base:
                clean_base = f"{clean_base}/v1"
            auth_header = token if str(token).startswith("Bearer ") else f"Bearer {token}"
            payload: Dict[str, Any] = {
                "fact": content.strip(),
                "category": cloud_category,
                "confidence": confidence,
            }
            if supersedes and supersedes.strip():
                payload["supersedes"] = supersedes.strip()
            if source_chat_id:
                payload["source_chat_id"] = source_chat_id

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{clean_base}/ai/memory/fact",
                        headers={"Authorization": auth_header, "Content-Type": "application/json"},
                        json=payload,
                    )
                    if resp.status_code in (200, 201):
                        local_res["cloud_synced"] = True
            except Exception:
                local_res["cloud_synced"] = False

        if local_res.get("status") == "success":
            try:
                from server import get_current_event_queue
                eq = get_current_event_queue()
                if eq is not None:
                    eq.put_nowait({
                        "event": "memory.updated",
                        "action": local_res.get("action"),
                        "target": local_res.get("target"),
                        "fact": local_res.get("fact"),
                    })
            except Exception:
                pass

        return local_res

    def format_for_system_prompt(
        self,
        existing_working_memory_text: Optional[str] = None,
        query_hint: Optional[str] = None,
        max_facts_per_category: int = 25,
        max_chars: int = 4000,
    ) -> str:
        """Formats memory facts as compact markdown for system prompt injection, enforcing token budgets and deduplicating against Turn 0 working memory."""
        data = self._read()
        parts = []

        try:
            from tools.valstorm_client import get_current_tenant_context
            ctx = get_current_tenant_context()
        except Exception:
            ctx = None

        wm_lower = (existing_working_memory_text or "").lower()
        skip_dup_user_facts = bool(
            (ctx and ctx.has_working_memory_injected) or bool(wm_lower)
        )

        header_str = "# Long-Term Memory (Persistent Across Sessions):\n"
        remaining_chars = max(max_chars - len(header_str), 200)

        user_facts = data.get("user", [])
        if skip_dup_user_facts and wm_lower:
            user_facts = [f for f in user_facts if f.strip().lower() not in wm_lower]
        elif ctx and ctx.is_cloud_mode and ctx.has_working_memory_injected and not wm_lower:
            user_facts = []

        mem_facts = data.get("memory", [])
        if wm_lower:
            mem_facts = [f for f in mem_facts if f.strip().lower() not in wm_lower]

        user_budget = remaining_chars // 2 if (user_facts and mem_facts) else remaining_chars
        selected_user = _select_budgeted_facts(
            user_facts,
            query_hint=query_hint,
            max_facts=max_facts_per_category,
            max_chars_budget=user_budget,
        )
        if selected_user:
            user_block = "## User Profile & Preferences:\n" + "\n".join(f"- {f}" for f in selected_user)
            parts.append(user_block)
            remaining_chars = max(remaining_chars - len(user_block), 150)

        selected_mem = _select_budgeted_facts(
            mem_facts,
            query_hint=query_hint,
            max_facts=max_facts_per_category,
            max_chars_budget=remaining_chars,
        )
        if selected_mem:
            parts.append("## Persistent Memory & Environment Notes:\n" + "\n".join(f"- {f}" for f in selected_mem))

        if not parts:
            return ""

        return header_str + "\n\n".join(parts)
