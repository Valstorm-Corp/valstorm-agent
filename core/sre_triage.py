"""SRE Triage Engine for Valstorm Agent Runtime.

Implements 3-way triage root-cause analysis, Slack Block Kit generation,
and on-demand bug ticketing for the Kubernetes SRE Agent.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("vsagent.sre_triage")

DEFAULT_ERROR_CHANNEL = os.getenv("SLACK_ALERT_CHANNEL", "C0BRH4P458E")
MGMT_ORG_ID = "org_dSnPMRjS1ZkkdYQ2"


class SRETriageResult(BaseModel):
    classification: str = Field(description="Client Error / Code Bug (P1/P2/P3) / Infrastructure Outage")
    category_type: str = Field(description="client_error | code_bug | infra_outage")
    severity: str = Field(default="P2", description="P1 | P2 | P3 | Info | Outage")
    offending_file: str = Field(default="unknown")
    offending_line: int = Field(default=0)
    root_cause: str = Field(description="Detailed explanation of why the failure occurred")
    proposed_fix: str = Field(description="Proposed code snippet or configuration correction")
    draft_ticket_title: str = Field(description="Structured title for B-xxx bug ticket")
    draft_ticket_description: str = Field(description="Markdown content for bug ticket")


def extract_code_snippet(file_path: str, line_number: int, context_lines: int = 10) -> str:
    """Reads lines from the local codebase surrounding the offending line number."""
    if not file_path or file_path == "unknown":
        return "<source file path unavailable>"

    p = Path(file_path)
    # If path is relative, resolve from current monorepo root or repo mount
    if not p.is_absolute():
        repo_root = Path(os.getenv("MONOREPO_ROOT", os.getcwd()))
        p = repo_root / p

    if not p.exists() or not p.is_file():
        return f"<file not found on disk: {file_path}>"

    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if total_lines == 0:
            return "<empty file>"

        target_idx = max(0, min(line_number - 1, total_lines - 1))
        start_idx = max(0, target_idx - context_lines)
        end_idx = min(total_lines, target_idx + context_lines + 1)

        snippet_lines = []
        for i in range(start_idx, end_idx):
            marker = ">>" if i == target_idx else "  "
            snippet_lines.append(f"{marker} {i + 1:4d} | {lines[i].rstrip()}")

        return "\n".join(snippet_lines)
    except Exception as e:
        return f"<failed to read source code: {e}>"


def classify_error_3way(
    exception_class: str,
    status_code: int,
    error_message: str,
    traceback_str: str,
    file_path: str,
    line_number: int,
    route: str,
) -> Tuple[str, str, str]:
    """Applies heuristic rule analysis to determine initial 3-way triage classification."""
    exc_lower = (exception_class or "").lower()
    msg_lower = (error_message or "").lower()
    tb_lower = (traceback_str or "").lower()

    # 1. Infrastructure / External Outages
    infra_keywords = [
        "connection refused", "serverselectiontimeouterror", "redistimeout",
        "event loop closed", "socket.gaierror", "pool exhaustion", "econnrefused",
        "503 service unavailable", "504 gateway timeout", "429 too many requests",
        "rate limit exceeded", "max retries exceeded", "closedconnectionerror"
    ]
    if any(k in msg_lower or k in tb_lower for k in infra_keywords):
        return "Infrastructure / External Outage", "infra_outage", "Outage"

    # 2. Client / User Input Errors
    client_keywords = [
        "validationerror", "badrequesterror", "400", "422", "404", "jsondecodeerror",
        "invalid custom sql", "permissiondeniederror", "unauthorized", "resourcenotfounderror"
    ]
    if status_code in [400, 401, 403, 404, 422] or any(k in exc_lower for k in client_keywords):
        return "Client Error (Bad Input / Schema Mismatch)", "client_error", "Info"

    # 3. Internal Code Bug
    # Determine severity
    is_p1 = any(k in route.lower() for k in ["/auth", "/billing", "/deploy", "/records/cud", "/login"]) or "security" in exc_lower
    severity = "P1" if is_p1 else "P2"
    return f"Code Bug ({severity})", "code_bug", severity


async def perform_sre_triage_analysis(
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
) -> SRETriageResult:
    """Executes root cause analysis using AST code inspection and Gemini LLM."""
    exception_class = payload.get("exception_class", "Exception")
    file_path = payload.get("file_path", "unknown")
    line_number = int(payload.get("line_number", 0)) if str(payload.get("line_number", 0)).isdigit() else 0
    route = payload.get("route", "/")
    method = payload.get("method", "GET")
    status_code = int(payload.get("status_code", 500))
    error_message = payload.get("error_message", "")
    traceback_str = payload.get("traceback", "")
    org_id = payload.get("org_id", "base")
    user_id = payload.get("user_id", "System")
    request_data = payload.get("request_data")

    # Read AST / source code snippet around offending line
    code_snippet = extract_code_snippet(file_path, line_number)

    classification, category_type, severity = classify_error_3way(
        exception_class=exception_class,
        status_code=status_code,
        error_message=error_message,
        traceback_str=traceback_str,
        file_path=file_path,
        line_number=line_number,
        route=route,
    )

    # Use Gemini model if API key is provided
    gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from providers.gemini import GeminiProvider
            from core.models import Message

            provider = GeminiProvider(api_key=gemini_key, default_model="gemini-flash-latest")
            system_prompt = (
                "You are the Valstorm Always-On Autonomous SRE Agent. "
                "Analyze the provided unhandled platform error, source code snippet, and execution traceback. "
                "Output strict valid JSON containing:\n"
                "- classification: 'Client Error (Bad Input / Schema Mismatch)' | 'Code Bug (P1)' | 'Code Bug (P2)' | 'Code Bug (P3)' | 'Infrastructure / External Outage'\n"
                "- category_type: 'client_error' | 'code_bug' | 'infra_outage'\n"
                "- severity: 'P1' | 'P2' | 'P3' | 'Info' | 'Outage'\n"
                "- offending_file: exact file path\n"
                "- offending_line: integer line number\n"
                "- root_cause: concise technical root cause explanation\n"
                "- proposed_fix: concrete Python/TypeScript code fix or mitigation\n"
                "- draft_ticket_title: Structured bug title e.g. '[Route/Module] Exception: Summary'\n"
                "- draft_ticket_description: Complete Markdown bug report with reproduction steps, root cause, and fix."
            )

            user_prompt = (
                f"Error Event Details:\n"
                f"- Route: {method} {route}\n"
                f"- Status Code: {status_code}\n"
                f"- Exception Class: {exception_class}\n"
                f"- Error Message: {error_message}\n"
                f"- File: {file_path}:{line_number}\n"
                f"- Org: {org_id} | User: {user_id}\n"
                f"- Request Data: {json.dumps(request_data, default=str)}\n\n"
                f"Traceback:\n```\n{traceback_str}\n```\n\n"
                f"Source Code Context:\n```python\n{code_snippet}\n```"
            )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]

            response = await provider.generate(messages=messages, temperature=0.1)
            raw_text = response.content.strip()
            # Clean markdown code block wrap if returned
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            parsed = json.loads(raw_text)
            return SRETriageResult(
                classification=parsed.get("classification", classification),
                category_type=parsed.get("category_type", category_type),
                severity=parsed.get("severity", severity),
                offending_file=parsed.get("offending_file", file_path),
                offending_line=int(parsed.get("offending_line", line_number)),
                root_cause=parsed.get("root_cause", error_message),
                proposed_fix=parsed.get("proposed_fix", "Inspect offending line for null/key checks."),
                draft_ticket_title=parsed.get("draft_ticket_title", f"[{route}] {exception_class}: {error_message}"),
                draft_ticket_description=parsed.get("draft_ticket_description", f"### Root Cause\n{error_message}\n\n### Traceback\n```\n{traceback_str}\n```"),
            )
        except Exception as llm_err:
            logger.warning(f"LLM triage enhancement failed ({llm_err}), falling back to deterministic heuristic triage.")

    # Deterministic fallback triage result
    return SRETriageResult(
        classification=classification,
        category_type=category_type,
        severity=severity,
        offending_file=file_path,
        offending_line=line_number,
        root_cause=f"{exception_class}: {error_message}" if error_message else "Unhandled platform exception",
        proposed_fix=f"Review logic at {file_path}:{line_number} and add defensive null/exception handling.",
        draft_ticket_title=f"[{route}] {exception_class}: {error_message[:60]}",
        draft_ticket_description=(
            f"## 1. Issue Summary\n"
            f"Unhandled `{exception_class}` triggered on `{method} {route}`.\n\n"
            f"## 2. Offending Location\n"
            f"- **File:** `{file_path}`\n"
            f"- **Line:** `{line_number}`\n\n"
            f"## 3. Traceback\n"
            f"```\n{traceback_str}\n```\n\n"
            f"## 4. Code Context\n"
            f"```python\n{code_snippet}\n```"
        ),
    )


def format_slack_block_kit(
    triage: SRETriageResult,
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Constructs official Slack Block Kit layout for the SRE triage report."""
    exception_class = payload.get("exception_class", "Exception")
    route = payload.get("route", "/")
    method = payload.get("method", "GET")
    org_id = payload.get("org_id", "base")
    user_id = payload.get("user_id", "System")
    count = payload.get("count", 1)

    count_badge = f" (Occurred {count}x in last 30m)" if count > 1 else ""

    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🚨 SRE Error Triage Report: {exception_class}{count_badge}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Triage Classification:*\n🔴 *{triage.classification}*"},
                {"type": "mrkdwn", "text": f"*Endpoint:*\n`{method} {route}`"},
                {"type": "mrkdwn", "text": f"*Offending Location:*\n`{triage.offending_file}:{triage.offending_line}`"},
                {"type": "mrkdwn", "text": f"*Affected Context:*\n`{org_id}` (`{user_id}`)"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Root Cause Analysis:*\n{triage.root_cause}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Proposed Fix / Draft:*\n```{triage.proposed_fix}```",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "💡 *To convert this diagnosis into an official ticket, reply in this thread:* `@Valstorm Bot create bug ticket`",
                }
            ],
        },
    ]


async def post_sre_report_to_slack(
    triage: SRETriageResult,
    payload: Dict[str, Any],
    slack_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Posts the formatted SRE Block Kit report into Slack channel #errors (C0BRH4P458E)."""
    channel = payload.get("channel_id", DEFAULT_ERROR_CHANNEL)
    token = slack_token or os.getenv("SLACK_TOKEN")
    blocks = format_slack_block_kit(triage, payload)
    fallback_text = f"🚨 SRE Triage: {triage.classification} on {payload.get('method')} {payload.get('route')}"

    if not token:
        logger.warning("SLACK_TOKEN not found. Printing Slack Block Kit to console instead.")
        return {"ok": True, "ts": f"mock_ts_{int(datetime.now().timestamp())}", "mock": True}

    auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
    headers = {"Authorization": auth_header, "Content-Type": "application/json; charset=utf-8"}
    req_body = {
        "channel": channel,
        "text": fallback_text,
        "blocks": blocks,
    }
    thread_ts = payload.get("slack_thread_ts") or payload.get("thread_ts")
    if thread_ts:
        req_body["thread_ts"] = thread_ts

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post("https://slack.com/api/chat.postMessage", json=req_body, headers=headers)
        return resp.json()


async def create_bug_ticket_in_db(
    triage_title: str,
    triage_description: str,
    priority: str = "Medium",
    mongo_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates an official B-xxx Bug Ticket in org_dSnPMRjS1ZkkdYQ2 upon human confirmation."""
    import motor.motor_asyncio

    mongo_url = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if not mongo_url and os.getenv("MONGO_HOST"):
        from urllib.parse import quote_plus

        user, password = os.getenv("MONGO_USER", ""), os.getenv("MONGO_PASSWORD", "")
        auth = f"{quote_plus(user)}:{quote_plus(password)}@" if user else ""
        mongo_url = f"mongodb://{auth}{os.getenv('MONGO_HOST')}:{os.getenv('MONGO_PORT', '27017')}"
    if mongo_client is None and not mongo_url:
        raise RuntimeError("Set MONGODB_URI (or MONGO_HOST/MONGO_USER/MONGO_PASSWORD) to create bug tickets")
    client = mongo_client or motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
    db = client[MGMT_ORG_ID]

    now = datetime.now(timezone.utc)

    # Find highest current ticket number to assign consecutive sequential number
    highest_ticket = await db.ticket.find_one(sort=[("number", -1)])
    next_number = (highest_ticket.get("number", 380) + 1) if highest_ticket else 381

    # Generate ticket base62 ID
    import uuid
    ticket_id = f"tick_{uuid.uuid4().hex[:16]}"

    formatted_name = f"B-{next_number} {triage_title}" if not triage_title.startswith("B-") else triage_title

    ticket_doc = {
        "id": ticket_id,
        "name": formatted_name,
        "number": next_number,
        "type": "Bug",
        "priority": "High" if priority == "P1" else ("Medium" if priority == "P2" else "Low"),
        "status": "Open",
        "description": triage_description,
        "description_markdown": triage_description,
        "created_date": now,
        "modified_date": now,
        "created_by": "user_system_sre",
        "modified_by": "user_system_sre",
        "owner": "user_system_sre",
    }

    await db.ticket.insert_one(ticket_doc)
    logger.info(f"Created Bug Ticket {formatted_name} ({ticket_id}) in {MGMT_ORG_ID}")

    return {
        "id": ticket_id,
        "number": next_number,
        "name": formatted_name,
        "url": f"valstorm://record/ticket/{ticket_id}/standard",
    }
