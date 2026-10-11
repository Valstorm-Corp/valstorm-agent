"""Declarative Fact Memory Store for Agent Runtime.

Maintains durable, high-signal facts across sessions:
- user: User profile, communication style, preferences
- memory: Environment facts, repository conventions, tool quirks, lessons learned
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

DEFAULT_MEMORY_FILE = Path.home() / ".valstorm" / "memories.json"


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

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        self.file_path = Path(file_path).expanduser().resolve() if file_path else DEFAULT_MEMORY_FILE
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

    def format_for_system_prompt(self) -> str:
        """Formats memory facts as compact markdown for system prompt injection."""
        data = self._read()
        parts = []

        user_facts = data.get("user", [])
        if user_facts:
            parts.append("## User Profile & Preferences:\n" + "\n".join(f"- {f}" for f in user_facts))

        mem_facts = data.get("memory", [])
        if mem_facts:
            parts.append("## Persistent Memory & Environment Notes:\n" + "\n".join(f"- {f}" for f in mem_facts))

        if not parts:
            return ""

        return "# Long-Term Memory (Persistent Across Sessions):\n" + "\n\n".join(parts)
