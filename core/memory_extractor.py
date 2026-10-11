"""Asynchronous Heuristic Context & Memory Extractor for Valstorm Agent Runtime.

Extracts high-signal declarative facts, user preferences, and repository rules
from conversation turns without latency impact on the user stream.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.memory import MemoryStore
from providers.base import BaseProvider

logger = logging.getLogger("valstorm.memory_extractor")

# High-precision regex heuristic patterns to gate model extraction (0 token overhead for 90%+ of turns)
MEMORY_TRIGGER_PATTERNS = [
    re.compile(r"\b(always|never|prefer|preference|preferences|make sure to|remember to|from now on|stop using|don't use|do not use)\b", re.IGNORECASE),
    re.compile(r"\b(i am|my name is|my role is|our team|we use|our stack|our company|my timezone|i work as)\b", re.IGNORECASE),
    re.compile(r"\b(actually,|no,|that's incorrect|that's wrong|we switched|switch from\b.+\bto\b|replaced\b.+\bwith\b)\b", re.IGNORECASE),
    re.compile(r"\b(our endpoint|the port is|we deployed|our convention|the rule is|standard is)\b", re.IGNORECASE),
    re.compile(r"\b(remember that|keep in mind that|note that for future)\b", re.IGNORECASE),
]

EXTRACTION_SYSTEM_PROMPT = """You are an ultra-fast, precise memory extraction sub-engine.
Analyze the user message and final assistant response to identify ANY durable, long-term facts, rules, or preferences.

Memory Categories:
- "user": Personal preferences, communication style, role, bio, preferred tools/patterns.
- "memory": Repository rules, architecture standards, environment facts, operational constraints, tech stack gotchas.

Rules:
1. ONLY extract durable facts that will be useful across future sessions. Ignore one-off, transient task instructions (e.g. "fix line 42", "run tests", "git status").
2. Formulate each fact as a concise, standalone declarative sentence (e.g., "Prefers 2-space indentation for TypeScript", "Transactional SMS provider is AWS SNS, Twilio is deprecated").
3. If a fact updates or negates an older convention, specify the "supersedes" keyword or phrase to clean up.
4. If NO long-term facts or preferences are present, return an empty JSON array `[]`.
5. Output MUST be valid JSON only matching the schema:
[
  {
    "category": "user" | "memory",
    "fact": "Concise declarative fact statement",
    "supersedes": "Optional search string of conflicting or outdated fact to remove/replace"
  }
]"""


def is_memory_candidate(user_input: Optional[str], assistant_output: Optional[str] = None) -> bool:
    """Evaluates whether the turn contains memory triggers using fast regex heuristics."""
    if not user_input or not user_input.strip():
        return False

    # Ignore explicit commands like /memorize or /help which are handled separately
    clean_input = user_input.strip()
    if clean_input.startswith("/") and not clean_input.startswith(("/remember", "/note")):
        return False

    # Check user input against triggers
    for pattern in MEMORY_TRIGGER_PATTERNS:
        if pattern.search(clean_input):
            return True

    return False


async def extract_memories_from_turn(
    user_input: str,
    assistant_output: str,
    provider: BaseProvider,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extracts structured declarative facts from a conversation turn via lightweight LLM call."""
    if not is_memory_candidate(user_input, assistant_output):
        return []

    # Prepare lightweight payload (clean text only, no raw tool logs or bloated JSON)
    prompt_text = (
        f"User Message:\n{user_input.strip()}\n\n"
        f"Assistant Final Answer:\n{(assistant_output or '').strip()[:1500]}"
    )

    try:
        from core.models import Message
        messages = [
            Message(role="system", content=EXTRACTION_SYSTEM_PROMPT),
            Message(role="user", content=prompt_text),
        ]

        target_model = model or "gemini-flash-latest"
        resp_msg = await provider.chat_complete(
            messages=messages,
            model=target_model,
            temperature=0.0,
            max_tokens=500,
        )

        content = resp_msg.content.strip()
        # Clean markdown codeblocks if model wrapped in ```json ... ```
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        if not content or content == "[]":
            return []

        parsed = json.loads(content)
        if isinstance(parsed, list):
            valid_facts = []
            for item in parsed:
                if isinstance(item, dict) and item.get("fact"):
                    cat = "user" if item.get("category") == "user" else "memory"
                    valid_facts.append({
                        "category": cat,
                        "fact": str(item["fact"]).strip(),
                        "supersedes": str(item.get("supersedes", "")).strip() or None,
                    })
            return valid_facts

    except Exception as e:
        logger.debug(f"Memory extraction skipped or encountered parsing error: {e}")

    return []


async def extract_and_commit_turn_memory(
    user_input: str,
    assistant_output: str,
    memory_store: MemoryStore,
    provider: BaseProvider,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Asynchronous background worker that extracts and saves facts into MemoryStore."""
    try:
        extracted = await extract_memories_from_turn(
            user_input=user_input,
            assistant_output=assistant_output,
            provider=provider,
            model=model,
        )

        results = []
        for item in extracted:
            cat = item["category"]
            fact_text = item["fact"]
            supersedes = item.get("supersedes")

            reconcile_res = memory_store.reconcile_fact(
                target=cat,
                content=fact_text,
                supersedes=supersedes,
            )
            results.append(reconcile_res)
            logger.info(f"[SmartMemory] Extracted and committed fact: {reconcile_res}")

        return results

    except Exception as e:
        logger.warning(f"[SmartMemory] Background extraction error: {e}")
        return []
