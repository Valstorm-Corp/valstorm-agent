"""Unit tests for Kubernetes SRE Agent, 3-way triage, and on-demand ticketing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.sre_triage import (
    classify_error_3way,
    extract_code_snippet,
    format_slack_block_kit,
    perform_sre_triage_analysis,
    SRETriageResult,
)


def test_classify_error_client_error():
    """Verify 4xx status codes and validation errors classify as Client Error."""
    cls_name, cat, sev = classify_error_3way(
        exception_class="ValidationError",
        status_code=422,
        error_message="Field 'deal_id' is required",
        traceback_str="fastapi.exceptions.RequestValidationError: ...",
        file_path="apps/api/app/records/routes.py",
        line_number=54,
        route="/v1/records/deal",
    )
    assert cat == "client_error"
    assert "Client Error" in cls_name
    assert sev == "Info"


def test_classify_error_code_bug():
    """Verify unhandled 5xx exceptions classify as Code Bug with appropriate severity."""
    cls_name, cat, sev = classify_error_3way(
        exception_class="KeyError",
        status_code=500,
        error_message="'user_id'",
        traceback_str="KeyError: 'user_id'\nFile 'apps/api/app/tasks/service.py', line 120",
        file_path="apps/api/app/tasks/service.py",
        line_number=120,
        route="/v1/tasks/bulk-assign",
    )
    assert cat == "code_bug"
    assert "Code Bug" in cls_name
    assert sev == "P2"


def test_classify_error_p1_critical_route():
    """Verify critical route failures classify as P1."""
    cls_name, cat, sev = classify_error_3way(
        exception_class="AttributeError",
        status_code=500,
        error_message="'NoneType' object has no attribute 'secret'",
        traceback_str="AttributeError: 'NoneType' object has no attribute 'secret'",
        file_path="apps/api/app/auth/auth_routes.py",
        line_number=88,
        route="/v1/auth/login",
    )
    assert cat == "code_bug"
    assert sev == "P1"


def test_classify_error_infra_outage():
    """Verify connection timeouts and provider errors classify as Infrastructure Outage."""
    cls_name, cat, sev = classify_error_3way(
        exception_class="ServerSelectionTimeoutError",
        status_code=500,
        error_message="No replica set members found yet, Timeout: 30s",
        traceback_str="pymongo.errors.ServerSelectionTimeoutError: No replica set members found yet",
        file_path="apps/api/app/valstorm/mongodb.py",
        line_number=45,
        route="/v1/records/lead",
    )
    assert cat == "infra_outage"
    assert "Infrastructure" in cls_name
    assert sev == "Outage"


def test_extract_code_snippet_nonexistent():
    """Verify extract_code_snippet gracefully handles missing files."""
    snippet = extract_code_snippet("nonexistent_file_path_xyz.py", 10)
    assert "<file not found" in snippet or "<source file" in snippet


def test_format_slack_block_kit():
    """Verify Block Kit message structure complies with Slack formatting standards."""
    triage = SRETriageResult(
        classification="Code Bug (P2)",
        category_type="code_bug",
        severity="P2",
        offending_file="apps/api/app/query/service.py",
        offending_line=142,
        root_cause="KeyError: 'sort_field' raised because the payload omitted optional sort parameters.",
        proposed_fix="sort_field = payload.get('sort_field', 'created_date')",
        draft_ticket_title="[Query] KeyError on missing sort_field",
        draft_ticket_description="### Root Cause\nMissing sort_field fallback.",
    )

    payload = {
        "exception_class": "KeyError",
        "route": "/v1/query/execute",
        "method": "POST",
        "org_id": "org_test123",
        "user_id": "user_test456",
        "count": 3,
    }

    blocks = format_slack_block_kit(triage, payload)
    assert len(blocks) >= 4
    assert blocks[0]["type"] == "header"
    assert "KeyError" in blocks[0]["text"]["text"]
    assert "Occurred 3x" in blocks[0]["text"]["text"]
    assert blocks[1]["type"] == "section"
    assert "Code Bug" in blocks[1]["fields"][0]["text"]
    assert "@Valstorm Bot create bug ticket" in blocks[-1]["elements"][0]["text"]


@pytest.mark.asyncio
async def test_perform_sre_triage_analysis_deterministic():
    """Verify fallback triage analysis works cleanly without external LLM API key."""
    payload = {
        "exception_class": "IndexError",
        "file_path": "apps/api/app/records/routes.py",
        "line_number": 42,
        "route": "/v1/records/list",
        "method": "GET",
        "status_code": 500,
        "error_message": "list index out of range",
        "traceback": "IndexError: list index out of range\n  File 'apps/api/app/records/routes.py', line 42",
        "org_id": "org_abc",
        "user_id": "user_def",
    }

    result = await perform_sre_triage_analysis(payload, api_key=None)
    assert isinstance(result, SRETriageResult)
    assert result.category_type == "code_bug"
    assert result.offending_line == 42
    assert "IndexError" in result.root_cause
