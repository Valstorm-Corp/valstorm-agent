"""Isolated Sandbox Valstorm Client & Craft Suite Environment for Agent Benchmarks.

Guarantees 100% offline isolation from production databases and external integrations:
1. `BenchmarkValstormClient`: Implements the `ValstormApiClient` interface against a
   per-task `valstorm_sandbox_db.json` file inside `workspace_dir`. Enforces Valstorm's
   data model guardrails (string lookup IDs vs. dict lookups, picklist tag labels vs. tag IDs,
   and `gatekeeper.py`-style AST validation for custom functions).
2. `provision_craft_suite_sandbox`: Installs deterministic CLI shims (`photocraft-cli`,
   `filmcraft-cli`, `effectcraft-cli`, `soundcraft-cli`, `vectorcraft-cli`, `deckcraft-cli`)
   and valid PNG preview frames into `workspace_dir` so Creative Director tasks can execute
   full CLI + `analyze_image` visual QA loops deterministically.
"""

from __future__ import annotations

import ast
import base64
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import uuid

# Minimal valid 1x1 PNG (transparent/dark slate pixel) for deterministic `analyze_image` QA
_MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

BANNED_AST_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "httpx", "urllib", "pickle"
})
BANNED_AST_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint"
})


def _default_sandbox_db() -> Dict[str, Any]:
    """Returns pre-seeded schemas and records for Office/CRM/VFS benchmark tasks."""
    return {
        "schemas": {
            "lead": {
                "api_name": "lead",
                "title": "Lead",
                "prefix": "lead",
                "fields": {
                    "name": {"type": "string", "required": True},
                    "status": {"type": "string", "format": "picklist", "options": ["New", "Active", "Qualified", "Closed Won", "Disqualified"]},
                    "loan_amount": {"type": "number", "format": "currency"},
                    "owner": {"type": "string", "format": "lookup", "target_schema": "user"},
                    "created_date": {"type": "string", "format": "datetime"},
                },
            },
            "contact": {
                "api_name": "contact",
                "title": "Contact",
                "prefix": "con",
                "fields": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string", "format": "phone"},
                    "company": {"type": "string"},
                    "status": {"type": "string", "format": "picklist", "options": ["Active", "Inactive", "Archived"]},
                    "owner": {"type": "string", "format": "lookup", "target_schema": "user"},
                },
            },
            "campaign": {
                "api_name": "campaign",
                "title": "Campaign",
                "prefix": "camp",
                "fields": {
                    "name": {"type": "string", "required": True},
                    "status": {"type": "string", "format": "picklist", "options": ["Draft", "Active", "Paused", "Completed"]},
                    "content_type": {"type": "string", "format": "picklist", "options": ["Carousel", "Single Image", "Video Reel", "Text Post"]},
                    "utm_source": {"type": "string", "format": "picklist", "options": ["linkedin", "meta", "youtube", "email"]},
                    "owner": {"type": "string", "format": "lookup", "target_schema": "user"},
                },
            },
            "task": {
                "api_name": "task",
                "title": "Task",
                "prefix": "task",
                "fields": {
                    "name": {"type": "string", "required": True},
                    "status": {"type": "string", "format": "picklist", "options": ["Todo", "In Progress", "Completed"]},
                    "campaign": {"type": "string", "format": "lookup", "target_schema": "campaign"},
                    "contact": {"type": "string", "format": "lookup", "target_schema": "contact"},
                    "owner": {"type": "string", "format": "lookup", "target_schema": "user"},
                },
            },
            "vault": {
                "api_name": "vault",
                "title": "Knowledge Vault",
                "prefix": "vaul",
                "fields": {
                    "name": {"type": "string", "required": True},
                    "parent_vault": {"type": "string", "format": "lookup", "target_schema": "vault"},
                },
            },
        },
        "records": {
            "lead": [
                {
                    "id": "lead_101",
                    "name": "Apex Logistics Expansion",
                    "status": "Active",
                    "loan_amount": 450000,
                    "owner": "user_qBulInDu18OcylWP",
                    "created_date": "2026-10-18T10:00:00Z",
                },
                {
                    "id": "lead_102",
                    "name": "Northstar Manufacturing Refi",
                    "status": "Active",
                    "loan_amount": 280000,
                    "owner": "user_qBulInDu18OcylWP",
                    "created_date": "2026-10-15T14:30:00Z",
                },
            ],
            "contact": [
                {
                    "id": "con_master_1",
                    "name": "Elena Rostova",
                    "email": "elena@acmecorp.com",
                    "phone": "+15550192831",
                    "company": "Acme Corp",
                    "status": "Active",
                    "owner": "user_qBulInDu18OcylWP",
                },
                {
                    "id": "con_dup_2",
                    "name": "Elena Rostova (Duplicate)",
                    "email": "e.rostova@acmecorp.com",
                    "phone": "+15550192831",
                    "company": "Acme Corp",
                    "status": "Active",
                    "owner": "user_qBulInDu18OcylWP",
                },
            ],
            "campaign": [],
            "task": [
                {
                    "id": "task_901",
                    "name": "Follow up on Acme Q4 renewal",
                    "status": "Todo",
                    "contact": "con_dup_2",
                    "owner": "user_qBulInDu18OcylWP",
                }
            ],
            "vault": [
                {
                    "id": "vaul_root_hub",
                    "name": "01_Company_Hub",
                    "parent_vault": None,
                }
            ],
            "list_filter": [],
            "drip_definition": [],
            "drip_enrollment": [],
            "function": [],
            "vfs_files": [],
        },
        "sql_queries_executed": [],
        "merges_executed": [],
        "functions_validated": [],
        "violations": [],
    }


class BenchmarkValstormClient:
    """100% local sandboxed Valstorm client for benchmarking Office, CRM, and VFS tasks safely."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.db_path = self.workspace_dir / "valstorm_sandbox_db.json"
        self.vfs_dir = self.workspace_dir / "vfs"
        self.vfs_dir.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._save_db(_default_sandbox_db())

    def _load_db(self) -> Dict[str, Any]:
        return json.loads(self.db_path.read_text(encoding="utf-8"))

    def _save_db(self, data: Dict[str, Any]) -> None:
        self.db_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def sql_query(self, query: str, bypass_cache: bool = False) -> Dict[str, Any]:
        db = self._load_db()
        clean_q = (query or "").strip()
        db["sql_queries_executed"].append(clean_q)

        # Match FROM <table_name>
        m = re.search(r"\bFROM\s+([a-zA-Z0-9_]+)", clean_q, re.IGNORECASE)
        table = m.group(1).lower() if m else "lead"
        rows = list(db["records"].get(table, []))

        # Simple filtering for contact company search (e.g. Acme Corp)
        if "acme" in clean_q.lower() and table == "contact":
            rows = [r for r in rows if "acme" in str(r.get("company", "")).lower() or "acme" in str(r.get("email", "")).lower()]

        self._save_db(db)
        return {"records": rows, "count": len(rows), "query": clean_q}

    async def schema_get(self, api_name: Optional[str] = None) -> Dict[str, Any]:
        db = self._load_db()
        schemas = db.get("schemas", {})
        if not api_name:
            return schemas
        key = api_name.strip().lower()
        return schemas.get(key, {"api_name": key, "fields": {}})

    def _validate_record_guardrails(
        self,
        db: Dict[str, Any],
        api_name: str,
        rec: Dict[str, Any],
    ) -> List[str]:
        """Checks Valstorm data model rules: plain string IDs for lookups and tag labels for picklists."""
        schema = db.get("schemas", {}).get(api_name, {})
        fields_spec = schema.get("fields", {})
        errors: List[str] = []

        for k, v in rec.items():
            f_spec = fields_spec.get(k, {})
            fmt = f_spec.get("format")
            # Rule 1: Never store dictionary lookups
            if isinstance(v, dict) and ("id" in v or "name" in v or fmt == "lookup"):
                msg = f"CRITICAL DATA MODEL VIOLATION on {api_name}.{k}: lookup fields must be plain string IDs, never dicts ({v})"
                errors.append(msg)
                db["violations"].append(msg)
            # Rule 2: Picklist fields must store human-readable tag labels, NEVER 'tag_...' IDs
            if fmt == "picklist" and isinstance(v, str) and v.strip().lower().startswith("tag_"):
                msg = f"CRITICAL DATA MODEL VIOLATION on {api_name}.{k}: picklist must use tag label, not tag ID ('{v}')"
                errors.append(msg)
                db["violations"].append(msg)

        return errors

    async def records_create(self, api_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        db = self._load_db()
        col_name = api_name.strip().lower()
        col = db["records"].setdefault(col_name, [])
        prefix = db.get("schemas", {}).get(col_name, {}).get("prefix", col_name[:4])

        created: List[Dict[str, Any]] = []
        all_errors: List[str] = []
        for idx, rec in enumerate(records, 1):
            errs = self._validate_record_guardrails(db, col_name, rec)
            all_errors.extend(errs)
            new_rec = dict(rec)
            if not new_rec.get("id"):
                new_rec["id"] = f"{prefix}_{len(col) + idx}_{uuid.uuid4().hex[:4]}"
            col.append(new_rec)
            created.append(new_rec)

        self._save_db(db)
        if all_errors:
            return {"status": "validation_warning", "errors": all_errors, "records": created}
        return {"status": "success", "records": created}

    async def records_update(self, api_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        db = self._load_db()
        col_name = api_name.strip().lower()
        col = db["records"].setdefault(col_name, [])
        updated: List[Dict[str, Any]] = []
        all_errors: List[str] = []

        for rec in records:
            errs = self._validate_record_guardrails(db, col_name, rec)
            all_errors.extend(errs)
            target_id = rec.get("id")
            for existing in col:
                if existing.get("id") == target_id:
                    existing.update(rec)
                    updated.append(existing)
                    break

        self._save_db(db)
        return {"status": "success" if not all_errors else "validation_warning", "errors": all_errors, "records": updated}

    async def records_delete(self, api_name: str, ids: List[str]) -> Dict[str, Any]:
        db = self._load_db()
        col_name = api_name.strip().lower()
        col = db["records"].get(col_name, [])
        before = len(col)
        db["records"][col_name] = [r for r in col if r.get("id") not in set(ids)]
        deleted_count = before - len(db["records"][col_name])
        self._save_db(db)
        return {"status": "success", "deleted_count": deleted_count}

    async def records_merge(
        self,
        master_id: str,
        duplicate_ids: List[str],
        schema_api_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = self._load_db()
        col_name = (schema_api_name or "contact").strip().lower()
        col = db["records"].get(col_name, [])
        dup_set = set(duplicate_ids)

        # Re-link child tasks pointing to duplicate_ids
        relinked_tasks = 0
        for t in db["records"].get("task", []):
            if t.get("contact") in dup_set:
                t["contact"] = master_id
                relinked_tasks += 1

        db["records"][col_name] = [r for r in col if r.get("id") not in dup_set]
        merge_entry = {
            "master_record_id": master_id,
            "duplicate_record_ids": list(duplicate_ids),
            "schema_api_name": col_name,
            "relinked_tasks": relinked_tasks,
        }
        db["merges_executed"].append(merge_entry)
        self._save_db(db)
        return {"status": "success", **merge_entry}

    async def vfs_write_file(
        self,
        name: str,
        content: str,
        vault_id: Optional[str] = None,
        file_id: Optional[str] = None,
        is_public: bool = False,
    ) -> Dict[str, Any]:
        db = self._load_db()
        clean_name = Path(name).name
        target_path = self.vfs_dir / clean_name
        target_path.write_text(content, encoding="utf-8")

        resolved_id = file_id or f"file_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": resolved_id,
            "name": clean_name,
            "vault_id": vault_id or "vaul_root_hub",
            "is_public": is_public,
            "size": len(content),
            "path": str(target_path.relative_to(self.workspace_dir)),
        }
        db["records"]["vfs_files"].append(entry)
        self._save_db(db)
        return {"status": "success", "file": entry}

    async def vfs_get_tree(self) -> Dict[str, Any]:
        db = self._load_db()
        return {"vaults": db["records"].get("vault", []), "files": db["records"].get("vfs_files", [])}

    async def vfs_browse_vault(self, vault_id: str = "root") -> Dict[str, Any]:
        db = self._load_db()
        return {
            "vault_id": vault_id,
            "sub_vaults": db["records"].get("vault", []),
            "files": [f for f in db["records"].get("vfs_files", []) if not vault_id or vault_id == "root" or f.get("vault_id") == vault_id],
        }

    async def vfs_browse_path(self, string_path: str) -> Dict[str, Any]:
        db = self._load_db()
        return {"path": string_path, "vaults": db["records"].get("vault", []), "files": db["records"].get("vfs_files", [])}

    async def vfs_get_file(self, file_id: str) -> Dict[str, Any]:
        db = self._load_db()
        for f in db["records"].get("vfs_files", []):
            if f.get("id") == file_id or f.get("name") == file_id:
                fpath = self.workspace_dir / f["path"]
                return {**f, "content": fpath.read_text(encoding="utf-8") if fpath.exists() else ""}
        return {"error": f"File '{file_id}' not found"}

    async def create_list_filter(
        self,
        name: str,
        object_api_name: str,
        sql_query: str,
        display_fields: Optional[List[str]] = None,
        is_pinned: bool = False,
        is_default: bool = False,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = self._load_db()
        lf = {
            "id": f"lf_{uuid.uuid4().hex[:8]}",
            "name": name,
            "object_api_name": object_api_name,
            "sql_query": sql_query,
            "display_fields": display_fields or [],
            "is_pinned": is_pinned,
            "is_default": is_default,
            "app_id": app_id,
        }
        db["records"]["list_filter"].append(lf)
        self._save_db(db)
        return {"status": "success", "list_filter": lf}

    async def function_validate(
        self,
        code: str,
        function_name: str = "test_function",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulates `gatekeeper.py` AST security validation for serverless functions."""
        db = self._load_db()
        ast_errors: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        if root_mod in BANNED_AST_MODULES:
                            ast_errors.append(f"Forbidden import '{alias.name}' blocked by gatekeeper.py")
                elif isinstance(node, ast.ImportFrom):
                    root_mod = (node.module or "").split(".")[0]
                    if root_mod in BANNED_AST_MODULES:
                        ast_errors.append(f"Forbidden import from '{node.module}' blocked by gatekeeper.py")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in BANNED_AST_BUILTINS:
                        ast_errors.append(f"Forbidden builtin '{node.func.id}()' blocked by gatekeeper.py")
        except SyntaxError as e:
            ast_errors.append(f"SyntaxError: {e}")

        passed = len(ast_errors) == 0
        fn_id = f"fun_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": fn_id,
            "function_name": function_name,
            "code": code,
            "valid": passed,
            "errors": ast_errors,
            "inputs": inputs or {},
        }
        db["functions_validated"].append(entry)
        if passed:
            db["records"]["function"].append(entry)
        self._save_db(db)
        return {"status": "valid" if passed else "invalid", "function_id": fn_id, **entry}

    async def create_drip_cadence(
        self,
        name: str,
        target_schema: str,
        steps: List[Dict[str, Any]],
        exit_conditions: Optional[Dict[str, Any]] = None,
        target_type: str = "function",
        target_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = self._load_db()
        drip = {
            "id": f"drip_{uuid.uuid4().hex[:8]}",
            "name": name,
            "target_schema": target_schema,
            "steps": steps,
            "exit_conditions": exit_conditions or {},
            "target_type": target_type,
            "target_id": target_id,
            "app_id": app_id,
        }
        db["records"]["drip_definition"].append(drip)
        self._save_db(db)
        return {"status": "success", "drip_definition": drip}


def provision_craft_suite_sandbox(workspace_dir: Path) -> None:
    """Provisions headless Craft Suite CLI shims and reference PNG assets inside `workspace_dir`."""
    wdir = Path(workspace_dir).resolve()
    bin_dir = wdir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed a valid PNG image so `analyze_image` always has a valid raster target
    for png_name in ("reference_plate.png", "check.png", "ad_preview.png", "reel_preview.png"):
        (wdir / png_name).write_bytes(_MINIMAL_PNG_BYTES)

    # Universal Craft Suite CLI shim that logs every invocation and generates requested output files
    shim_code = """#!/usr/bin/env python3
import json
import pathlib
import base64
import sys

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

def main():
    cli_name = pathlib.Path(sys.argv[0]).name
    args = sys.argv[1:]
    wdir = pathlib.Path.cwd()
    log_path = wdir / "craft_cli_invocations.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cli": cli_name, "args": args}) + "\\n")

    # Materialize any output file arguments (--out, --save, --save-as, export, convert)
    for i, a in enumerate(args):
        if a in ("--out", "--save", "--save-as") and i + 1 < len(args):
            target = wdir / args[i + 1]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix.lower() == ".png":
                target.write_bytes(PNG_BYTES)
            elif not target.exists():
                target.write_text(json.dumps({"cli": cli_name, "args": args}, indent=2), encoding="utf-8")
        elif a.endswith((".png", ".psd", ".fcproj", ".ecproj", ".scraft", ".fcpxml", ".xml", ".otio", ".json", ".wav", ".mp4", ".mov")):
            target = wdir / a
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix.lower() == ".png":
                target.write_bytes(PNG_BYTES)
            elif not target.exists():
                target.write_text(json.dumps({"cli": cli_name, "args": args}, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "cli": cli_name, "args": args}))

if __name__ == "__main__":
    main()
"""
    for tool_bin in (
        "photocraft-cli",
        "filmcraft-cli",
        "effectcraft-cli",
        "soundcraft-cli",
        "vectorcraft-cli",
        "deckcraft-cli",
        "figcraft-cli",
        "gridcraft-cli",
    ):
        script_path = bin_dir / tool_bin
        script_path.write_text(shim_code, encoding="utf-8")
        script_path.chmod(0o755)
