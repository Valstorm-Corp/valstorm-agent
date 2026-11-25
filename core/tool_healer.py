"""System-1 Tool Argument Self-Healer (Priority 6 / FI-566).

Resolves minor schema, field, object API name, lookup dict, and file path mismatches in <2ms
using System-1 (`ChoiceQuestion`) + lexical similarity before a tool call fails with `is_error=True`
and triggers an expensive full-context `reasoning_effort='high'` retry turn.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from core.decision_model import ChoiceQuestion, SystemOneClient

# Canonical Valstorm object API name aliases & plurals
_OBJECT_API_NAME_SYNONYMS: Dict[str, str] = {
    "leads": "lead",
    "contacts": "contact",
    "campaigns": "campaign",
    "tasks": "task",
    "deals": "deal",
    "opportunities": "opportunity",
    "accounts": "account",
    "companies": "account",
    "company": "account",
    "organizations": "account",
    "tickets": "ticket",
    "vaults": "vault",
    "knowledge_vault": "vault",
    "knowledge_vaults": "vault",
    "files": "file",
    "users": "user",
}

# Canonical field-level synonyms for common CRM/Office objects
_FIELD_SYNONYM_HINTS: Dict[str, str] = {
    "campaign_name": "name",
    "lead_name": "name",
    "contact_name": "name",
    "task_name": "name",
    "vault_name": "name",
    "filename": "name",
    "file_name": "name",
    "title": "name",
    "amount": "loan_amount",
    "loan_value": "loan_amount",
    "deal_amount": "loan_amount",
    "utm": "utm_source",
    "source": "utm_source",
    "channel": "utm_source",
    "type": "content_type",
    "post_type": "content_type",
    "format": "content_type",
    "owner_id": "owner",
    "assigned_to": "owner",
    "assignee": "owner",
    "parent": "parent_vault",
    "parent_id": "parent_vault",
    "email_address": "email",
    "phone_number": "phone",
}


def heal_object_api_name_s1(
    api_name: str,
    available_schemas: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    """Heals pluralized, synonym, or slightly misspelled object `api_name` values in <1ms."""
    raw = (api_name or "").strip()
    if not raw:
        return raw, False
    low = raw.lower()

    if available_schemas and low in available_schemas:
        return low, low != raw

    if low in _OBJECT_API_NAME_SYNONYMS:
        target = _OBJECT_API_NAME_SYNONYMS[low]
        if not available_schemas or target in available_schemas:
            return target, True

    if low.endswith("s") and len(low) > 3:
        singular = low[:-1]
        if not available_schemas or singular in available_schemas:
            return singular, True

    if available_schemas:
        candidates = list(available_schemas.keys())
        close = difflib.get_close_matches(low, candidates, n=1, cutoff=0.72)
        if close:
            return close[0], True

    return low, low != raw


def heal_field_name_s1(
    field_name: str,
    valid_fields: Dict[str, Any],
) -> Tuple[str, bool]:
    """Uses System-1 ChoiceQuestion + lexical similarity to heal a mismatched field name against valid schema fields."""
    if not field_name or not valid_fields or field_name in valid_fields:
        return field_name, False

    low = field_name.strip().lower()
    if low in valid_fields:
        return low, True

    # 1. Check canonical field synonym hints first
    hint_target = _FIELD_SYNONYM_HINTS.get(low)
    if hint_target and hint_target in valid_fields:
        return hint_target, True

    # 2. Check prefix/suffix stripping (e.g. `campaign_status` -> `status`, `owner_id` -> `owner`)
    for vf in valid_fields:
        if low.endswith(f"_{vf}") or low == f"{vf}_id" or low == f"{vf}_name":
            return vf, True

    # 3. System-1 ChoiceQuestion + SequenceMatcher over valid schema fields
    criteria: Dict[str, str] = {}
    for vf, spec in valid_fields.items():
        fmt = spec.get("format", "") if isinstance(spec, dict) else ""
        ftype = spec.get("type", "") if isinstance(spec, dict) else ""
        tokens = vf.replace("_", " ")
        criteria[vf] = f"{vf} {tokens} {fmt} {ftype}"

    s1 = SystemOneClient(backend="local")
    resp = s1.evaluate(
        state={"user_input": low.replace("_", " "), "goal": low},
        questions={
            "field_match": ChoiceQuestion(
                instructions="Select the matching schema field API name.",
                criteria=criteria,
            )
        },
    )
    ans = resp.get_choice("field_match")
    best_lex = difflib.get_close_matches(low, list(valid_fields.keys()), n=1, cutoff=0.62)
    if best_lex:
        return best_lex[0], True
    if ans.choice in valid_fields and ans.confidence >= 0.65:
        return ans.choice, True

    return field_name, False


def _extract_schema_fields(schema_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts field definitions from either Open Valstorm (`properties`) or legacy (`fields`) schema dicts."""
    if not isinstance(schema_spec, dict):
        return {}
    props = schema_spec.get("properties")
    if isinstance(props, dict) and props:
        return dict(props)
    fields = schema_spec.get("fields")
    if isinstance(fields, dict) and fields:
        return dict(fields)
    return {}


def _normalize_schema_map(available_schemas: Any) -> Dict[str, Any]:
    """Normalizes schema collections (dict, {'schemas': [...]}, or list) into {api_name: schema_dict}."""
    if isinstance(available_schemas, dict):
        if "schemas" in available_schemas and isinstance(available_schemas["schemas"], list):
            available_schemas = available_schemas["schemas"]
        else:
            return {str(k).lower(): v for k, v in available_schemas.items() if isinstance(v, dict)}
    if isinstance(available_schemas, list):
        return {
            str(s.get("api_name") or "").lower(): s
            for s in available_schemas
            if isinstance(s, dict) and s.get("api_name")
        }
    return {}


def heal_record_payload_s1(
    record: Dict[str, Any],
    schema_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Heals a single record dictionary for `valstorm_record_cud`:
    1. Heals mismatched field names against `schema_spec['properties']` / `schema_spec['fields']`.
    2. Unwraps dictionary lookups (`{'id': 'user_123'}` -> `'user_123'`) so lookup guardrails never fail.
    3. Normalizes case-insensitive picklist option labels (e.g. `'carousel'` -> `'Carousel'`).
    """
    if not isinstance(record, dict):
        return record, []

    valid_fields = _extract_schema_fields(schema_spec)
    healed: Dict[str, Any] = {}
    healed_notes: List[str] = []

    for k, v in record.items():
        target_k = k
        if valid_fields and k not in valid_fields:
            cand_k, was_healed = heal_field_name_s1(k, valid_fields)
            if was_healed:
                target_k = cand_k
                healed_notes.append(f"field '{k}'->'{target_k}'")

        f_spec = valid_fields.get(target_k, {}) if isinstance(valid_fields, dict) else {}
        fmt = f_spec.get("format") if isinstance(f_spec, dict) else None

        # Heal dictionary lookup objects into plain string IDs
        target_v = v
        if isinstance(v, dict) and ("id" in v or fmt == "lookup"):
            extracted_id = v.get("id") or v.get("_id")
            if isinstance(extracted_id, str) and extracted_id.strip():
                target_v = extracted_id.strip()
                healed_notes.append(f"lookup_dict '{target_k}'->'{target_v}'")

        # Heal picklist case / tag_ prefix if options are known in schema
        if fmt == "picklist" and isinstance(target_v, str):
            options = f_spec.get("options") or []
            clean_v = target_v.strip()
            if clean_v.lower().startswith("tag_"):
                stem = clean_v[4:].split("_")[0].lower()
                for opt in options:
                    if opt.lower().startswith(stem) or stem in opt.lower():
                        target_v = opt
                        healed_notes.append(f"picklist_tag '{clean_v}'->'{opt}'")
                        break
            else:
                for opt in options:
                    if opt.lower() == clean_v.lower() and opt != clean_v:
                        target_v = opt
                        healed_notes.append(f"picklist_case '{clean_v}'->'{opt}'")
                        break

        healed[target_k] = target_v

    return healed, healed_notes


_SQL_FROM_RE = re.compile(r"\bFROM\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
_SQL_SELECT_RE = re.compile(r"^\s*SELECT\s+(.*?)\s+FROM\s+", re.IGNORECASE | re.DOTALL)


def heal_sql_query_s1(
    query: str,
    available_schemas: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """Heals table names (`FROM leads` -> `FROM lead`) and SELECT/WHERE column typos in `valstorm_sql_query`."""
    clean_q = (query or "").strip()
    if not clean_q:
        return clean_q, []

    notes: List[str] = []
    m_from = _SQL_FROM_RE.search(clean_q)
    if not m_from:
        return clean_q, notes

    norm_schemas = _normalize_schema_map(available_schemas)
    raw_table = m_from.group(1)
    healed_table, table_changed = heal_object_api_name_s1(raw_table, norm_schemas or None)
    if table_changed and healed_table != raw_table:
        clean_q = re.sub(
            rf"\bFROM\s+{re.escape(raw_table)}\b",
            f"FROM {healed_table}",
            clean_q,
            count=1,
            flags=re.IGNORECASE,
        )
        notes.append(f"table '{raw_table}'->'{healed_table}'")

    schema_spec = norm_schemas.get(healed_table.lower()) or {}
    valid_fields = _extract_schema_fields(schema_spec)
    if valid_fields:
        valid_fields["id"] = {"type": "string"}
        # Heal column identifiers in SELECT and WHERE/ORDER BY clauses
        sql_keywords = {
            "select", "from", "where", "and", "or", "not", "in", "like", "ilike",
            "order", "by", "group", "having", "as", "is", "asc", "desc", "limit",
            "offset", "distinct", "case", "when", "then", "else", "end", "count",
            "sum", "avg", "min", "max", "coalesce", "cast", "join", "left", "inner",
            "on", "me", "today", "yesterday", "this_week", "this_month", "this_quarter",
            "this_year", "null", "true", "false",
        }
        for col_cand in set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", clean_q)):
            low_col = col_cand.lower()
            if (
                low_col in sql_keywords
                or low_col == healed_table
                or col_cand in valid_fields
                or low_col.startswith(("last_", "next_", "month_", "week_", "con_", "lead_", "user_", "camp_", "vaul_"))
            ):
                continue
            # Only heal if col_cand is a known field synonym or close typo of a valid field
            if low_col in _FIELD_SYNONYM_HINTS or difflib.get_close_matches(low_col, list(valid_fields.keys()), n=1, cutoff=0.76):
                healed_col, col_changed = heal_field_name_s1(col_cand, valid_fields)
                if col_changed and healed_col != col_cand:
                    clean_q = re.sub(rf"\b{re.escape(col_cand)}\b", healed_col, clean_q)
                    notes.append(f"column '{col_cand}'->'{healed_col}'")

    return clean_q, notes


def heal_file_path_s1(path: str, base_dir: Path) -> Tuple[str, bool]:
    """If `path` does not exist directly under `base_dir`, searches workspace files for a unique basename/fuzzy match."""
    raw = (path or "").strip()
    if not raw:
        return raw, False

    candidate = (base_dir / raw).resolve()
    if candidate.is_file():
        return raw, False

    target_name = Path(raw).name.lower()
    target_stem = Path(raw).stem.lower()
    ignored = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build"}

    exact_basename_hits: List[str] = []
    all_rel_files: List[str] = []

    try:
        for root, dirs, files in base_dir.walk() if hasattr(base_dir, "walk") else _walk_compat(base_dir):
            dirs[:] = [d for d in dirs if d not in ignored and not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                rel = str((Path(root) / fname).relative_to(base_dir))
                all_rel_files.append(rel)
                if fname.lower() == target_name:
                    exact_basename_hits.append(rel)
    except Exception:
        return raw, False

    if len(exact_basename_hits) == 1:
        return exact_basename_hits[0], True

    if len(exact_basename_hits) > 1:
        # Score multiple basename matches via System-1 against original path tokens
        criteria = {rel: rel.replace("/", " ").replace("_", " ") for rel in exact_basename_hits[:10]}
        s1 = SystemOneClient(backend="local")
        resp = s1.evaluate(
            state={"user_input": raw.replace("/", " "), "goal": raw},
            questions={"path_match": ChoiceQuestion(instructions="Select best matching file path", criteria=criteria)},
        )
        ans = resp.get_choice("path_match")
        if ans.choice in criteria and ans.confidence >= 0.60:
            return ans.choice, True
        return exact_basename_hits[0], True

    # Fuzzy match on filename if the LLM made a 1-2 character typo (e.g. `src/poool.py` -> `src/pool.py`)
    by_name = {Path(rel).name.lower(): rel for rel in all_rel_files}
    close_names = difflib.get_close_matches(target_name, list(by_name.keys()), n=1, cutoff=0.75)
    if close_names:
        return by_name[close_names[0]], True

    # Check stem match with same extension
    ext = Path(raw).suffix.lower()
    same_ext = [rel for rel in all_rel_files if Path(rel).suffix.lower() == ext and target_stem in Path(rel).stem.lower()]
    if len(same_ext) == 1:
        return same_ext[0], True

    return raw, False


def _walk_compat(base_dir: Path):
    import os
    for root, dirs, files in os.walk(base_dir):
        yield Path(root), dirs, files
