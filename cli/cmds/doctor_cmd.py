"""`vsagent doctor`: verify every failover tier is usable with the credential the runtime would really use.

For each tier of the cascade it reports where the key came from (masked, never the key itself), whether the
model ID exists in the provider's catalog (DigitalOcean), and - unless --offline - sends one tiny request so a
bad key, wrong model ID or overloaded backend is found *before* a failover needs it.
"""

import asyncio
import inspect
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx
import typer
from rich.console import Console
from rich.table import Table

from core.keystore import KeyStore, mask_key
from core.models import Message, ToolCall, ToolResult
from core.retry import extract_status_code
from providers import build_fallback_chain, infer_cascade_tier
from providers.fallback import FallbackTier

from ..helpers import ensure_provider_key

console = Console()

PROBE_PROMPT = "Reply with the single word: ok"
_DO_PROVIDER_NAMES = ("do", "digitalocean")


def _short_error(exc: BaseException) -> str:
    """One-line, secret-free description of a provider error."""
    status = extract_status_code(exc)
    message = " ".join(str(exc).split())
    prefix = type(exc).__name__ + (f" {status}" if status else "")
    return f"{prefix}: {message[:100]}"


def _tidy_source(source: Optional[str]) -> str:
    """Shortens a key source for display: ~ for the home dir, and only the tail of long paths."""
    if not source:
        return "unknown source"
    home = str(Path.home())
    if source.startswith(home):
        source = "~" + source[len(home):]
    return source if len(source) <= 48 else "…" + source[-47:]


def _backend_of(tier: FallbackTier) -> Optional[str]:
    """Which backend a Gemini tier really talks to (vertex / aistudio / valstorm-passthrough), if the provider says."""
    label = getattr(tier.provider, "backend_label", None)
    if callable(label):
        try:
            return str(label())
        except Exception:
            return None
    return None


def _fetch_do_catalog(base_url: str, api_key: str, timeout: float) -> Tuple[Optional[Set[str]], Optional[str]]:
    """Returns (model ids, None) or (None, error). The catalog endpoint requires auth, so this also validates the key."""
    url = base_url.rstrip("/") + "/models"
    try:
        resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    except Exception as exc:
        return None, _short_error(exc)
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {' '.join(resp.text.split())[:100]}"
    try:
        return {m["id"] for m in resp.json().get("data", [])}, None
    except Exception as exc:
        return None, f"unreadable catalog: {_short_error(exc)}"


async def _probe(tier: FallbackTier, timeout: float) -> Tuple[bool, str, float]:
    """Sends one tiny request through the tier's provider exactly as a real turn would."""
    started = time.monotonic()
    try:
        res = tier.provider.generate(
            messages=[Message(role="user", content=PROBE_PROMPT)],
            tools=None,
            model=tier.model,
        )
        if inspect.isawaitable(res):
            res = await asyncio.wait_for(res, timeout)
        msg = res[0] if isinstance(res, tuple) else res
        text = (getattr(msg, "content", None) or "").strip()
        detail = f'replied "{text[:30]}"' if text else "empty reply (reasoning model may have spent its token budget)"
        return True, detail, time.monotonic() - started
    except asyncio.TimeoutError:
        return False, f"timeout after {timeout:.0f}s", time.monotonic() - started
    except Exception as exc:
        return False, _short_error(exc), time.monotonic() - started


REPLAY_TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }
]
_GEMINI_FAMILY_NAMES = ("gemini", "valstorm")


def _replay_messages(target_is_gemini: bool) -> List[Message]:
    """A mid-conversation history that is about to hop to a *different* model family.

    Gemini targets get a history whose tool call came from a non-Gemini model (no thought signature, which the
    Gemini provider must render as text). Every other target gets a Gemini-originated tool call carrying a
    thought signature, which the OpenAI-compatible provider must not forward to a strict endpoint.
    """
    call = ToolCall(id="call_replay_1", name="add", arguments={"a": 17, "b": 25})
    if target_is_gemini:
        origin = {"provider": "do", "model": "openai-gpt-oss-120b"}
    else:
        call.thought_signature = b"replay-signature-bytes"
        origin = {"provider": "gemini", "model": "gemini-flash-latest"}
    return [
        Message(role="user", content="What is 17 + 25? Use the add tool, then tell me the answer."),
        Message(role="assistant", content=None, tool_calls=[call], **origin),
        Message(
            role="tool",
            content="42",
            tool_result=ToolResult(call_id=call.id, name="add", output=42),
        ),
    ]


async def _probe_replay(tier: FallbackTier, timeout: float) -> Tuple[bool, str, float]:
    """Replays a tool-call conversation (tool call + tool result) through the tier and expects the answer, 42."""
    started = time.monotonic()
    try:
        res = tier.provider.generate(
            messages=_replay_messages(tier.provider_name in _GEMINI_FAMILY_NAMES),
            tools=REPLAY_TOOLS,
            model=tier.model,
        )
        if inspect.isawaitable(res):
            res = await asyncio.wait_for(res, timeout)
        msg = res[0] if isinstance(res, tuple) else res
        text = (getattr(msg, "content", None) or "").strip()
        elapsed = time.monotonic() - started
        if "42" in text:
            return True, f'replay answered "{text[:30]}"', elapsed
        if getattr(msg, "tool_calls", None):
            return False, "replay: model called the tool again instead of answering", elapsed
        if not text:
            return False, "replay: empty reply (reasoning model may have spent its token budget)", elapsed
        return False, f'replay: unexpected answer "{text[:40]}"', elapsed
    except asyncio.TimeoutError:
        return False, f"replay: timeout after {timeout:.0f}s", time.monotonic() - started
    except Exception as exc:
        return False, f"replay: {_short_error(exc)}", time.monotonic() - started


async def _replay_all(tiers: List[FallbackTier], timeout: float) -> List[Tuple[bool, str, float]]:
    return [await _probe_replay(t, timeout) for t in tiers]


async def _run_checks(
    tiers: List[FallbackTier], timeout: float, replay: bool
) -> Tuple[List[Tuple[bool, str, float]], List[Optional[Tuple[bool, str, float]]]]:
    """Runs the probe (and optionally the replay) for every tier inside ONE event loop.

    Provider SDK clients are cached on the provider and keep connections bound to the loop that first used them, so
    running the probe and the replay in two separate asyncio.run() calls fails the second with "Event loop is closed".
    """
    probes = await _probe_all(tiers, timeout)
    replays: List[Optional[Tuple[bool, str, float]]] = [None] * len(tiers)
    if replay:
        replays = list(await _replay_all(tiers, timeout))
    return list(probes), replays


async def _probe_all(tiers: List[FallbackTier], timeout: float) -> List[Tuple[bool, str, float]]:
    # Sequential on purpose: a doctor run must not trip rate limits it is trying to diagnose.
    return [await _probe(t, timeout) for t in tiers]


def doctor_command(
    provider: str = typer.Option("valstorm", "--provider", help="Primary provider the cascade is built around."),
    model: str = typer.Option("gemini-flash-latest", "--model", "-m", help="Primary model (also selects the cascade tier)."),
    tier: Optional[str] = typer.Option(None, "--tier", help="Cascade tier: reasoner, thinker or worker (default: inferred from --model)."),
    live: bool = typer.Option(True, "--live/--offline", help="Send one tiny request per tier (a few tokens each)."),
    timeout: float = typer.Option(60.0, "--timeout", help="Per-tier request timeout in seconds."),
    replay: bool = typer.Option(
        False,
        "--replay/--no-replay",
        help="Also replay a tool-call conversation through every tier (checks cross-model handoff; a few more tokens each).",
    ),
):
    """Check every failover tier: key source, model ID validity and a live probe."""
    primary = provider.strip().lower()
    cascade_tier = (tier or infer_cascade_tier(model_name=model)).strip().lower()

    api_key: Optional[str] = None
    if primary:
        try:
            api_key = ensure_provider_key(primary, interactive=False)
        except SystemExit:
            console.print(f"[yellow]Primary provider '{primary}' is not authenticated; checking backup tiers only.[/yellow]\n")
            primary = ""

    chain = build_fallback_chain(
        tier=cascade_tier,
        primary_provider=primary or None,
        primary_model=model if primary else None,
        keystore=KeyStore(),
        api_key=api_key,
    )
    tiers = list(chain.tiers)

    # DigitalOcean catalog check (one request per distinct key/base URL).
    catalogs: Dict[Tuple[str, str], Tuple[Optional[Set[str]], Optional[str]]] = {}
    for t in tiers:
        if t.provider_name in _DO_PROVIDER_NAMES:
            base = getattr(t.provider, "base_url", None) or "https://inference.do-ai.run/v1"
            key = getattr(t.provider, "api_key", None) or ""
            if (base, key) not in catalogs:
                catalogs[(base, key)] = _fetch_do_catalog(base, key, min(timeout, 30.0))

    probes: List[Optional[Tuple[bool, str, float]]] = [None] * len(tiers)
    replays: List[Optional[Tuple[bool, str, float]]] = [None] * len(tiers)
    if live:
        probes, replays = asyncio.run(_run_checks(tiers, timeout, replay))

    table = Table(title=f"vsagent doctor - {cascade_tier} cascade", show_lines=False)
    columns = ["#", "Tier", "Model", "Key (source)", "Catalog", "Live"]
    if live and replay:
        columns.append("Replay")
    columns.append("Detail")
    for column in columns:
        table.add_column(column, overflow="fold")

    healthy = 0
    for position, (t, probe, replay_probe) in enumerate(zip(tiers, probes, replays), 1):
        key_value = getattr(t.provider, "api_key", None)
        key_cell = f"{mask_key(key_value)}\n[dim]{_tidy_source(t.key_source)}[/dim]"
        backend = _backend_of(t)
        model_cell = f"{t.model}\n[dim]backend: {backend}[/dim]" if backend else t.model

        catalog_cell, catalog_ok, detail = "n/a", True, ""
        if t.provider_name in _DO_PROVIDER_NAMES:
            base = getattr(t.provider, "base_url", None) or "https://inference.do-ai.run/v1"
            ids, err = catalogs[(base, key_value or "")]
            if ids is None:
                catalog_cell, catalog_ok, detail = "[red]auth/catalog error[/red]", False, err or ""
            elif t.model in ids:
                catalog_cell = "[green]ok[/green]"
            else:
                catalog_cell, catalog_ok = "[red]MISSING[/red]", False
                detail = f"'{t.model}' is not in the DO catalog"

        if probe is None:
            live_cell = "[dim]skipped[/dim]"
            tier_ok = catalog_ok
        else:
            probe_ok, probe_detail, seconds = probe
            live_cell = f"[green]ok[/green] {seconds:.1f}s" if probe_ok else f"[red]FAIL[/red] {seconds:.1f}s"
            detail = probe_detail if not detail else f"{detail}; {probe_detail}"
            tier_ok = catalog_ok and probe_ok

        row = [str(position), t.label, model_cell, key_cell, catalog_cell, live_cell]
        if live and replay:
            if replay_probe is None:
                row.append("[dim]skipped[/dim]")
            else:
                replay_ok, replay_detail, replay_seconds = replay_probe
                row.append(f"[green]ok[/green] {replay_seconds:.1f}s" if replay_ok else f"[red]FAIL[/red] {replay_seconds:.1f}s")
                if not replay_ok:
                    detail = replay_detail if not detail else f"{detail}; {replay_detail}"
                tier_ok = tier_ok and replay_ok

        healthy += 1 if tier_ok else 0
        row.append(detail)
        table.add_row(*row)

    console.print(table)
    mode = "live-checked" if live else "key/catalog-checked only (use --live for a real request)"
    if live and replay:
        mode += " + tool-call replay"
    console.print(f"\n{healthy}/{len(tiers)} tiers healthy ({mode}).")
    if healthy == 0:
        console.print("[bold red]No usable tier: a request would fail.[/bold red]")
        raise typer.Exit(code=1)
    if healthy == 1:
        console.print("[yellow]Only one usable tier: there is no failover redundancy right now.[/yellow]")
