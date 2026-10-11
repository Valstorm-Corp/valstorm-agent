"""Always-On Kubernetes SRE Agent & Worker Daemon.

Runs on port 8660. Consumes diagnostic tasks from Redis queue (valstorm:queue:sre),
executes 3-way triage root cause analysis via AST code inspection, posts reports to Slack #errors (C0BRH4P458E),
and provides on-demand Bug Ticket creation (B-xxx) in org_dSnPMRjS1ZkkdYQ2 upon Slack confirmation.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from core.resp_redis import AsyncRespRedisClient
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.sre_triage import (
    DEFAULT_ERROR_CHANNEL,
    MGMT_ORG_ID,
    SRETriageResult,
    create_bug_ticket_in_db,
    perform_sre_triage_analysis,
    post_sre_report_to_slack,
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
SRE_PORT = int(os.getenv("SRE_AGENT_PORT", "8660"))

# State variables
_running = True
_worker_task: Optional[asyncio.Task] = None


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


async def parse_and_process_task(raw_task: Any, redis_client: Any) -> bool:
    """Parses task bytes from Redis/Asynq and executes triage pipeline."""
    try:
        payload = None
        # Handle Asynq Protobuf or raw JSON
        if isinstance(raw_task, bytes):
            try:
                # Try raw json
                payload = json.loads(raw_task.decode("utf-8"))
            except Exception:
                try:
                    # Try asynq protobuf message extraction
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
                # If raw_task is an Asynq task_id (UUID string), resolve from hash
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

        logger.info(
            f"Processing SRE diagnostic task for {payload.get('method')} {payload.get('route')} "
            f"({payload.get('exception_class')}) at {payload.get('file_path')}:{payload.get('line_number')}"
        )

        # 1. Perform 3-Way Triage RCA
        triage_result: SRETriageResult = await perform_sre_triage_analysis(payload)

        # 2. Post Block Kit Report to Slack #errors
        slack_resp = await post_sre_report_to_slack(triage_result, payload)

        # 3. Store Slack thread_ts in Redis under sre_thread:{fingerprint} for thread replies & dedupe updates
        fingerprint = payload.get("fingerprint")
        thread_ts = payload.get("slack_thread_ts") or slack_resp.get("ts")
        if fingerprint and thread_ts and redis_client:
            await redis_client.set(f"sre_thread:{fingerprint}", thread_ts, ex=1800)
            # Store latest triage result for on-demand ticket creation
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
    """Background consumer loop pulling tasks from Redis queue."""
    logger.info(f"SRE Agent Queue Worker started. Listening on Redis queues...")
    while _running:
        try:
            r = AsyncRespRedisClient(REDIS_URL)
            async with r:
                while _running:
                    # Pop from Asynq pending list or direct list
                    task_data = None
                    try:
                        # Check direct queue first
                        raw = await r.rpop(SRE_QUEUE_KEY)
                        if raw:
                            task_data = raw
                        else:
                            # Check Asynq pending queue layout
                            raw = await r.rpop(ASYNQ_SRE_PENDING)
                            if raw:
                                # Dereference task_id from Asynq hash
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



@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task, _running
    _running = True
    _worker_task = asyncio.create_task(sre_queue_worker_loop())
    logger.info("🚀 SRE Agent pod initialized and ready.")
    yield
    _running = False
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    logger.info("🛑 SRE Agent pod gracefully terminated.")


app = FastAPI(
    title="Valstorm SRE Autonomous Agent",
    description="Always-on Kubernetes SRE Agent for automated error triage and diagnostic reporting.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    return {
        "status": "ok",
        "service": "valstorm-sre-agent",
        "version": "1.0.0",
        "queue": SRE_QUEUE_KEY,
        "org": MGMT_ORG_ID,
    }


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

        # Reply to Slack thread if thread_ts is supplied
        if req.slack_thread_ts:
            token = os.getenv("SLACK_TOKEN")
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


@app.post("/v1/sre/slack-events")
async def slack_events_webhook(request: Request):
    """Handles Slack event subscriptions and mentions (@Valstorm Bot create bug ticket)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "invalid_json"}, status_code=400)

    # 1. Slack URL verification challenge
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    event_type = event.get("type")
    text = (event.get("text") or "").lower()
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")

    # 2. Check for human user asking to create bug ticket
    if event_type in ["app_mention", "message"] and "create bug ticket" in text:
        logger.info(f"Received Slack on-demand ticket command in channel {channel}, thread {thread_ts}")

        # Attempt to retrieve cached triage result for this thread if available
        # Or parse summary from request
        title = "Platform Exception Auto-Triage"
        description = f"Reported from Slack thread {thread_ts} by user {event.get('user')}"

        try:
            ticket = await create_bug_ticket_in_db(
                triage_title=title,
                triage_description=description,
                priority="Medium",
            )
            # Reply in Slack thread
            token = os.getenv("SLACK_TOKEN")
            if token:
                auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
                reply_body = {
                    "channel": channel,
                    "thread_ts": thread_ts,
                    "text": f"✅ Created official Bug Ticket *{ticket['name']}* (`{ticket['id']}`)\n🔗 *<{ticket['url']}|View Ticket in Valstorm>*",
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        "https://slack.com/api/chat.postMessage",
                        json=reply_body,
                        headers={"Authorization": auth_header, "Content-Type": "application/json"},
                    )
            return {"ok": True, "ticket_id": ticket["id"]}
        except Exception as e:
            logger.error(f"Error handling Slack ticket command: {e}")
            return {"ok": False, "error": str(e)}

    return {"ok": True, "ignored": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SRE_PORT)
