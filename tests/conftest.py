"""Pytest configuration and live telemetry summary reporter for agent runtime tests."""

import time
import pytest

LIVE_TELEMETRY_LOG = []


def record_live_telemetry(test_name: str, provider: str, model: str, tools_used: list, prompt_tokens: int, completion_tokens: int, duration_sec: float):
    LIVE_TELEMETRY_LOG.append({
        "test": test_name,
        "provider": provider,
        "model": model,
        "tools": ", ".join(tools_used) if tools_used else "-",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "duration": f"{duration_sec:.2f}s",
    })


def pytest_sessionfinish(session, exitstatus):
    """Prints a formatted telemetry summary table if any live tests executed."""
    if not LIVE_TELEMETRY_LOG:
        return

    print("\n\n" + "═" * 105)
    print("                      ⚡ AGENT RUNTIME LIVE TEST TELEMETRY REPORT ⚡")
    print("═" * 105)
    header = f"{'Test Case':<32} | {'Provider':<10} | {'Model':<22} | {'Tools':<14} | {'In/Out':<10} | {'Total':<6} | {'Time':<6}"
    print(header)
    print("─" * 105)
    
    total_in = 0
    total_out = 0
    total_tokens_all = 0

    for item in LIVE_TELEMETRY_LOG:
        row = f"{item['test']:<32} | {item['provider']:<10} | {item['model']:<22} | {item['tools']:<14} | {item['prompt_tokens']}/{item['completion_tokens']:<6} | {item['total_tokens']:<6} | {item['duration']:<6}"
        print(row)
        total_in += item['prompt_tokens']
        total_out += item['completion_tokens']
        total_tokens_all += item['total_tokens']

    print("─" * 105)
    print(f"{'TOTALS':<32} | {'ALL':<10} | {'-':<22} | {'-':<14} | {total_in}/{total_out:<6} | {total_tokens_all:<6} |")
    print("═" * 105 + "\n")
