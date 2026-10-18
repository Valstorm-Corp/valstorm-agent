"""Always-On Kubernetes SRE & Autonomous Growth Agent Daemon.

Runs on port 8660.
1. SRE Triage Engine:
   - Consumes diagnostic tasks from Redis queue (valstorm:queue:sre)
   - Executes 3-way triage root cause analysis via AST code inspection
   - Posts reports to Slack #errors (C0BRH4P458E) and creates B-xxx Bug Tickets on demand.
2. Autonomous Growth & Marketing Telemetry Engine:
   - Harvests multi-horizon metrics (Yesterday, 7d, 30d, All-Time) across Google Ads, GA4, GSC, and Instantly
   - Runs a bounded ReAct investigation loop (`core.growth_agent`) across all verticals (Greenfield + CT Local/Contractors)
   - Uploads HTML/Markdown scorecards to Valstorm Cloud VFS and posts digest + threaded ReAct recommendations to #marketing (C0C8WBKV8PJ)
   - Supports interactive Slack thread follow-ups (`@Valstorm Bot create ticket A1` or `@Valstorm Bot <question>`)
"""

import asyncio
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from core.resp_redis import AsyncRespRedisClient
from core.sre_triage import (
    DEFAULT_ERROR_CHANNEL,
    MGMT_ORG_ID,
    SRETriageResult,
    create_bug_ticket_in_db,
    perform_sre_triage_analysis,
    post_sre_report_to_slack,
)
from core.growth_agent import (
    DEFAULT_MARKETING_CHANNEL,
    answer_growth_slack_question,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("ValstormSREAgent")


def resolve_redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD")
    user = os.getenv("REDIS_USER", "default")
    env = os.getenv("ENV", "local")
    scheme = "rediss" if (env in ["production", "dev"] or port == 25061) else "redis"
    if password:
        auth = f"{user}:{password}@" if user else f":{password}@"
    else:
        auth = ""
    return f"{scheme}://{auth}{host}:{port}/0"


REDIS_URL = resolve_redis_url()
SRE_QUEUE_KEY = "valstorm:queue:sre"
ASYNQ_SRE_PENDING = f"asynq:{{{SRE_QUEUE_KEY}}}:pending"
GROWTH_QUEUE_KEY = "valstorm:queue:growth"
SRE_PORT = int(os.getenv("SRE_AGENT_PORT", "8660"))

GROWTH_CRON_ENABLED = os.getenv("GROWTH_CRON_ENABLED", "true").lower() in ("true", "1", "yes")
GROWTH_DIGEST_HOUR_UTC = int(os.getenv("GROWTH_DIGEST_HOUR_UTC", "13"))

# State variables
_running = True
_worker_task: Optional[asyncio.Task] = None
_growth_cron_task: Optional[asyncio.Task] = None
_latest_growth_cache: Dict[str, Any] = {}


class SREDirectTriageRequest(BaseModel):
    exception_class: str = "Exception"
    file_path: str = "unknown"
    line_number: int = 0
    route: str = "/"
    method: str = "GET"
    status_code: int = 500
    error_message: str = ""
    traceback: str = ""
    org_id: str = "base"
    user_id: str = "System"
    request_data: Optional[Any] = None
    channel_id: Optional[str] = DEFAULT_ERROR_CHANNEL


class SRECreateTicketRequest(BaseModel):
    title: str
    description: str
    priority: str = "Medium"
    slack_thread_ts: Optional[str] = None
    slack_channel: Optional[str] = DEFAULT_ERROR_CHANNEL


class GrowthDigestRunRequest(BaseModel):
    send_slack: bool = True
    upload_vfs: bool = True
    days: int = 7
    run_agent: bool = True


async def execute_growth_digest_pipeline(
    send_slack: bool = True,
    upload_vfs: bool = True,
    days: int = 7,
    run_agent: bool = True,
    redis_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Loads TelemetryHarvester from scripting/marketing and executes the multi-horizon + ReAct digest."""
    global _latest_growth_cache

    monorepo_root = Path(os.getenv("MONOREPO_ROOT", Path(__file__).resolve().parents[2]))
    marketing_dir = monorepo_root / "scripting" / "marketing"
    if str(marketing_dir) not in sys.path and marketing_dir.exists():
        sys.path.insert(0, str(marketing_dir))

    from unified_telemetry_reporter import TelemetryHarvester

    def _sync_run() -> Dict[str, Any]:
        harvester = TelemetryHarvester()
        return harvester.run(
            send_slack=send_slack,
            upload_vfs=upload_vfs,
            days=days,
            run_agent=run_agent,
        )

    result = await asyncio.to_thread(_sync_run)
    _latest_growth_cache = result

    # Cache in Redis for interactive Slack thread Q&A and proposal ticketing
    slack_ts = result.get("slack_ts")
    cache_payload = json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "slack_ts": slack_ts,
            "slack_channel": result.get("slack_channel", DEFAULT_MARKETING_CHANNEL),
            "agent_review": result.get("agent_review"),
            "telemetry_snapshot": result.get("telemetry_snapshot"),
            "vfs_html": result.get("vfs_html"),
            "vfs_md": result.get("vfs_md"),
        },
        default=str,
    )

    try:
        if redis_client:
            await redis_client.set("growth_snapshot:latest", cache_payload, ex=86400 * 7)
            if slack_ts:
                await redis_client.set(f"growth_thread:{slack_ts}", cache_payload, ex=86400 * 7)
        else:
            r = AsyncRespRedisClient(REDIS_URL)
            async with r:
                await r.set("growth_snapshot:latest", cache_payload, ex=86400 * 7)
                if slack_ts:
                    await r.set(f"growth_thread:{slack_ts}", cache_payload, ex=86400 * 7)
    except Exception as cache_err:
        logger.debug(f"Could not persist growth snapshot to Redis: {cache_err}")

    return result


async def parse_and_process_task(raw_task: Any, redis_client: Any) -> bool:
    """Parses task bytes from Redis/Asynq and executes triage pipeline."""
    try:
        payload = None
        if isinstance(raw_task, bytes):
            try:
                payload = json.loads(raw_task.decode("utf-8"))
            except Exception:
                try:
                    from valstorm.asynq_pb2 import TaskMessage
                    msg = TaskMessage()
                    msg.ParseFromString(raw_task)
                    payload = json.loads(msg.payload.decode("utf-8"))
                except Exception as pb_err:
                    logger.warning(f"Could not parse task bytes: {pb_err}")
                    return False
        elif isinstance(raw_task, str):
            try:
                payload = json.loads(raw_task)
            except Exception:
                if redis_client and ("-" in raw_task or len(raw_task) >= 20):
                    task_key = f"asynq:{{{SRE_QUEUE_KEY}}}:t:{raw_task.strip()}"
                    try:
                        raw_msg = await redis_client.hget(task_key, "msg")
                        if raw_msg:
                            try:
                                from valstorm.asynq_pb2 import TaskMessage
                                msg = TaskMessage()
                                msg.ParseFromString(raw_msg.encode("latin1") if isinstance(raw_msg, str) else raw_msg)
                                payload = json.loads(msg.payload.decode("utf-8"))
                            except Exception:
                                payload = json.loads(raw_msg)
                    except Exception as hget_err:
                        logger.warning(f"Could not resolve Asynq task hash for {raw_task}: {hget_err}")
        elif isinstance(raw_task, dict):
            payload = raw_task

        if not payload:
            return False

        # Check if task is a growth telemetry run request
        if payload.get("task_type") == "growth_digest":
            logger.info("Executing queued Growth & Telemetry Digest task...")
            await execute_growth_digest_pipeline(
                send_slack=payload.get("send_slack", True),
                upload_vfs=payload.get("upload_vfs", True),
                days=int(payload.get("days", 7)),
                run_agent=payload.get("run_agent", True),
                redis_client=redis_client,
            )
            return True

        logger.info(
            f"Processing SRE diagnostic task for {payload.get('method')} {payload.get('route')} "
            f"({payload.get('exception_class')}) at {payload.get('file_path')}:{payload.get('line_number')}"
        )

        # 1. Perform 3-Way Triage RCA
        triage_result: SRETriageResult = await perform_sre_triage_analysis(payload)

        # 2. Post Block Kit Report to Slack #errors
        slack_resp = await post_sre_report_to_slack(triage_result, payload)

        # 3. Store Slack thread_ts in Redis under sre_thread:{fingerprint}
        fingerprint = payload.get("fingerprint")
        thread_ts = payload.get("slack_thread_ts") or slack_resp.get("ts")
        if fingerprint and thread_ts and redis_client:
            await redis_client.set(f"sre_thread:{fingerprint}", thread_ts, ex=1800)
            await redis_client.set(
                f"sre_triage_cached:{fingerprint}",
                triage_result.model_dump_json(),
                ex=3600,
            )

        logger.info(
            f"Successfully triaged {fingerprint}: {triage_result.classification} "
            f"(Severity: {triage_result.severity}, Slack TS: {thread_ts})"
        )
        return True
    except Exception as e:
        logger.error(f"Error processing SRE task: {e}", exc_info=True)
        return False


async def sre_queue_worker_loop():
    """Background consumer loop pulling tasks from SRE and Growth Redis queues."""
    logger.info("SRE & Growth Agent Queue Worker started. Listening on Redis queues...")
    while _running:
        try:
            r = AsyncRespRedisClient(REDIS_URL)
            async with r:
                while _running:
                    task_data = None
                    growth_raw = None
                    try:
                        # 1. Check Growth queue
                        growth_raw = await r.rpop(GROWTH_QUEUE_KEY)
                        if growth_raw:
                            g_payload = {"task_type": "growth_digest"}
                            if isinstance(growth_raw, (bytes, str)):
                                try:
                                    parsed_g = json.loads(growth_raw.decode("utf-8") if isinstance(growth_raw, bytes) else growth_raw)
                                    if isinstance(parsed_g, dict):
                                        g_payload.update(parsed_g)
                                except Exception:
                                    pass
                            await parse_and_process_task(g_payload, r)
                            continue

                        # 2. Check direct SRE queue
                        raw = await r.rpop(SRE_QUEUE_KEY)
                        if raw:
                            task_data = raw
                        else:
                            # Check Asynq pending queue layout
                            raw = await r.rpop(ASYNQ_SRE_PENDING)
                            if raw:
                                if isinstance(raw, str) and not raw.strip().startswith("{"):
                                    task_key = f"asynq:{{{SRE_QUEUE_KEY}}}:t:{raw.strip()}"
                                    try:
                                        msg_val = await r.hget(task_key, "msg")
                                        task_data = msg_val or raw
                                    except Exception as h_lookup_err:
                                        logger.warning(f"Asynq hash dereference failed: {h_lookup_err}")
                                        task_data = raw
                                else:
                                    task_data = raw
                    except Exception as pop_err:
                        logger.debug(f"Redis pop check: {pop_err}")

                    if task_data:
                        await parse_and_process_task(task_data, r)
                    else:
                        await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as conn_err:
            logger.warning(f"Redis connection retry in 3s: {conn_err}")
            await asyncio.sleep(3.0)


async def growth_daily_scheduler_loop():
    """Built-in daily UTC cron scheduler so marketing reports run in K8s without laptop dependency."""
    logger.info(
        f"Growth Daily Scheduler active (enabled={GROWTH_CRON_ENABLED}, target_hour_utc={GROWTH_DIGEST_HOUR_UTC}:00)."
    )
    last_ran_date: Optional[str] = None
    while _running:
        try:
            if GROWTH_CRON_ENABLED:
                now_utc = datetime.now(timezone.utc)
                today_key = now_utc.strftime("%Y-%m-%d")
                if now_utc.hour == GROWTH_DIGEST_HOUR_UTC and last_ran_date != today_key:
                    already_ran = False
                    redis_marker = f"growth_digest_ran:{today_key}"
                    try:
                        r = AsyncRespRedisClient(REDIS_URL)
                        async with r:
                            existing = await r.get(redis_marker)
                            if existing:
                                already_ran = True
                            else:
                                await r.set(redis_marker, "1", ex=86400)
                    except Exception:
                        pass

                    last_ran_date = today_key
                    if not already_ran:
                        logger.info(f"⏰ Triggering scheduled daily Growth & Telemetry Digest for {today_key}...")
                        await execute_growth_digest_pipeline(send_slack=True, upload_vfs=True, days=7, run_agent=True)
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            break
        except Exception as sched_err:
            logger.error(f"Error in growth daily scheduler loop: {sched_err}")
            await asyncio.sleep(60.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task, _growth_cron_task, _running
    _running = True
    _worker_task = asyncio.create_task(sre_queue_worker_loop())
    _growth_cron_task = asyncio.create_task(growth_daily_scheduler_loop())
    logger.info("🚀 SRE & Autonomous Growth Agent pod initialized and ready.")
    yield
    _running = False
    for t in (_worker_task, _growth_cron_task):
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    logger.info("🛑 SRE & Autonomous Growth Agent pod gracefully terminated.")


app = FastAPI(
    title="Valstorm SRE & Autonomous Growth Agent",
    description="Always-on Kubernetes Agent for automated SRE error triage and ReAct Marketing Growth Telemetry.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    return {
        "status": "ok",
        "service": "valstorm-sre-agent",
        "version": "1.1.0",
        "queues": [SRE_QUEUE_KEY, GROWTH_QUEUE_KEY],
        "growth_cron_enabled": GROWTH_CRON_ENABLED,
        "growth_digest_hour_utc": GROWTH_DIGEST_HOUR_UTC,
        "org": MGMT_ORG_ID,
    }


@app.post("/v1/growth/run-digest")
async def run_growth_digest_endpoint(req: GrowthDigestRunRequest):
    """Triggers a full multi-horizon marketing telemetry pull, ReAct agent review, VFS upload, and Slack digest."""
    try:
        result = await execute_growth_digest_pipeline(
            send_slack=req.send_slack,
            upload_vfs=req.upload_vfs,
            days=req.days,
            run_agent=req.run_agent,
        )
        return {
            "ok": True,
            "status": result.get("status"),
            "slack_ts": result.get("slack_ts"),
            "vfs_html": result.get("vfs_html"),
            "vfs_md": result.get("vfs_md"),
            "active_ads_campaigns_count": result.get("active_ads_campaigns_count"),
            "paused_ads_campaigns_count": result.get("paused_ads_campaigns_count"),
            "active_meta_campaigns_count": result.get("active_meta_campaigns_count"),
            "paused_meta_campaigns_count": result.get("paused_meta_campaigns_count"),
            "active_outbound_campaigns_count": result.get("active_outbound_campaigns_count"),
            "paused_outbound_campaigns_count": result.get("paused_outbound_campaigns_count"),
            "agent_review": result.get("agent_review"),
        }
    except Exception as e:
        logger.error(f"Failed to run growth digest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/growth/latest")
async def get_latest_growth_digest():
    """Returns the most recent cached multi-horizon telemetry and ReAct agent review."""
    if _latest_growth_cache:
        return {
            "ok": True,
            "source": "memory",
            "agent_review": _latest_growth_cache.get("agent_review"),
            "vfs_html": _latest_growth_cache.get("vfs_html"),
            "vfs_md": _latest_growth_cache.get("vfs_md"),
        }
    try:
        r = AsyncRespRedisClient(REDIS_URL)
        async with r:
            raw = await r.get("growth_snapshot:latest")
            if raw:
                data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                return {"ok": True, "source": "redis", **data}
    except Exception:
        pass
    return {"ok": False, "message": "No growth digest cached yet."}


@app.post("/v1/sre/triage")
async def direct_triage_endpoint(req: SREDirectTriageRequest):
    """Direct HTTP endpoint to trigger SRE triage analysis on an error event."""
    payload = req.model_dump()
    triage_result = await perform_sre_triage_analysis(payload)
    slack_resp = await post_sre_report_to_slack(triage_result, payload)
    return {
        "ok": True,
        "triage": triage_result.model_dump(),
        "slack": slack_resp,
    }


@app.post("/v1/sre/create-ticket")
async def create_ticket_endpoint(req: SRECreateTicketRequest):
    """Creates a B-xxx bug ticket in org_dSnPMRjS1ZkkdYQ2 and replies to the Slack thread."""
    try:
        ticket = await create_bug_ticket_in_db(
            triage_title=req.title,
            triage_description=req.description,
            priority=req.priority,
        )

        if req.slack_thread_ts:
            token = os.getenv("SLACK_TOKEN") or os.getenv("SLACK_BOT_TOKEN")
            if token:
                auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
                reply_body = {
                    "channel": req.slack_channel or DEFAULT_ERROR_CHANNEL,
                    "thread_ts": req.slack_thread_ts,
                    "text": f"✅ Created official Bug Ticket *{ticket['name']}* ({ticket['id']})\n🔗 *<{ticket['url']}|View Ticket in Valstorm>*",
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        "https://slack.com/api/chat.postMessage",
                        json=reply_body,
                        headers={"Authorization": auth_header, "Content-Type": "application/json"},
                    )

        return {"ok": True, "ticket": ticket}
    except Exception as e:
        logger.error(f"Failed to create bug ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _load_cached_growth_context(thread_ts: Optional[str]) -> Optional[Dict[str, Any]]:
    """Loads cached growth snapshot for a Slack thread or falls back to latest."""
    try:
        r = AsyncRespRedisClient(REDIS_URL)
        async with r:
            if thread_ts:
                raw_t = await r.get(f"growth_thread:{thread_ts}")
                if raw_t:
                    return json.loads(raw_t.decode("utf-8") if isinstance(raw_t, bytes) else raw_t)
            raw_latest = await r.get("growth_snapshot:latest")
            if raw_latest:
                return json.loads(raw_latest.decode("utf-8") if isinstance(raw_latest, bytes) else raw_latest)
    except Exception:
        pass
    if _latest_growth_cache:
        return _latest_growth_cache
    return None


async def _post_slack_thread_reply(channel: str, thread_ts: str, text: str) -> None:
    token = os.getenv("SLACK_TOKEN") or os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return
    auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            "https://slack.com/api/chat.postMessage",
            json={"channel": channel, "thread_ts": thread_ts, "text": text},
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
        )


@app.post("/v1/sre/slack-events")
@app.post("/v1/growth/slack-events")
async def slack_events_webhook(request: Request):
    """Handles Slack event subscriptions and mentions for both SRE (#errors) and Growth (#marketing)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "invalid_json"}, status_code=400)

    # 1. Slack URL verification challenge
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return {"ok": True, "ignored": True}

    event_type = event.get("type")
    raw_text = event.get("text") or ""
    text = raw_text.lower()
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")

    if event_type not in ["app_mention", "message"]:
        return {"ok": True, "ignored": True}

    # 2. Check for Growth Proposal Ticket Creation: e.g. "@Valstorm Bot create ticket A1"
    proposal_match = re.search(r"create\s+(?:bug\s+)?ticket\s+(?:for\s+)?\[?(a\d+)\]?", text, re.IGNORECASE)
    if proposal_match:
        action_id = proposal_match.group(1).upper()
        growth_ctx = await _load_cached_growth_context(thread_ts)
        proposals = ((growth_ctx or {}).get("agent_review") or {}).get("proposals") or []
        matched_prop = next((p for p in proposals if str(p.get("action_id", "")).upper() == action_id), None)

        if matched_prop:
            title = f"[Growth {action_id}] {matched_prop.get('title')}"
            description = (
                f"## Growth Action Proposal `{action_id}` ({matched_prop.get('vertical')})\n"
                f"- **Category:** `{matched_prop.get('category')}`\n"
                f"- **Priority:** `{matched_prop.get('priority')}`\n\n"
                f"### Telemetry Rationale\n{matched_prop.get('rationale')}\n\n"
                f"### Suggested Execution\n{matched_prop.get('suggested_execution')}"
            )
            priority = matched_prop.get("priority", "P2")
        else:
            title = f"[Growth {action_id}] Marketing Telemetry Action Item"
            description = f"Created from Slack growth thread {thread_ts} by user {event.get('user')}."
            priority = "Medium"

        try:
            ticket = await create_bug_ticket_in_db(
                triage_title=title,
                triage_description=description,
                priority=priority,
            )
            await _post_slack_thread_reply(
                channel,
                thread_ts,
                f"✅ Created Growth Action Ticket *{ticket['name']}* (`{ticket['id']}`)\n🔗 *<{ticket['url']}|View Ticket in Valstorm>*",
            )
            return {"ok": True, "ticket_id": ticket["id"], "action_id": action_id}
        except Exception as e:
            logger.error(f"Error creating growth ticket: {e}")
            return {"ok": False, "error": str(e)}

    # 3. Check for standard SRE bug ticket command: "@Valstorm Bot create bug ticket"
    if "create bug ticket" in text:
        logger.info(f"Received Slack on-demand ticket command in channel {channel}, thread {thread_ts}")
        title = "Platform Exception Auto-Triage"
        description = f"Reported from Slack thread {thread_ts} by user {event.get('user')}"

        try:
            ticket = await create_bug_ticket_in_db(
                triage_title=title,
                triage_description=description,
                priority="Medium",
            )
            await _post_slack_thread_reply(
                channel,
                thread_ts,
                f"✅ Created official Bug Ticket *{ticket['name']}* (`{ticket['id']}`)\n🔗 *<{ticket['url']}|View Ticket in Valstorm>*",
            )
            return {"ok": True, "ticket_id": ticket["id"]}
        except Exception as e:
            logger.error(f"Error handling Slack ticket command: {e}")
            return {"ok": False, "error": str(e)}

    # 4. Interactive Growth ReAct Q&A in #marketing or on a growth digest thread
    is_marketing_channel = channel == os.getenv("SLACK_MARKETING_CHANNEL", DEFAULT_MARKETING_CHANNEL)
    if event_type == "app_mention" and (is_marketing_channel or "growth" in text or "ads" in text or "instantly" in text or "ga4" in text):
        growth_ctx = await _load_cached_growth_context(thread_ts)
        snapshot = (growth_ctx or {}).get("telemetry_snapshot") or {}
        clean_question = re.sub(r"<@[^>]+>", "", raw_text).strip()
        if clean_question:
            answer = await answer_growth_slack_question(clean_question, snapshot)
            await _post_slack_thread_reply(channel, thread_ts, answer)
            return {"ok": True, "replied": True}

    return {"ok": True, "ignored": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SRE_PORT)
