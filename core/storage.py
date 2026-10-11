"""SQLite WAL-backed Session Storage mirroring Valstorm ai_chat and ai_chat_message schemas.

Provides durable, atomic persistence for agent sessions, message histories,
token usage ledgers, and FTS5 full-text search across historical turns.
"""

import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.models import Message, SessionState, ToolCall, ToolResult, UsageMetadata

DEFAULT_DB_PATH = Path.home() / ".valstorm" / "agent_state.db"


def _json_serial_default(o: Any) -> Any:
    """Handles bytes and other non-JSON types for safe serialization."""
    if isinstance(o, bytes):
        return {"__bytes__": base64.b64encode(o).decode("ascii")}
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_serial_default)


def _restore_bytes_recursive(data: Any) -> Any:
    """Restores {"__bytes__": ...} dicts to raw bytes."""
    if isinstance(data, dict):
        if len(data) == 1 and "__bytes__" in data:
            try:
                return base64.b64decode(data["__bytes__"])
            except Exception:
                return data["__bytes__"]
        return {k: _restore_bytes_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_restore_bytes_recursive(item) for item in data]
    return data


class SessionStore:
    """Manages SQLite storage for ai_chat and ai_chat_message tables with FTS5 search."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path).expanduser().resolve() if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite connection configured with WAL mode and foreign keys."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Initializes tables and FTS5 index matching Valstorm schemas."""
        with self._get_connection() as conn:
            # 1. ai_chat table (mirrors ai_chat.json)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_chat (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'Active',
                    created_date TEXT NOT NULL,
                    modified_date TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_input_tokens INTEGER DEFAULT 0,
                    total_output_tokens INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}'
                );
                """
            )

            # 2. ai_chat_message table (mirrors ai_chat_message.json)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_chat_message (
                    id TEXT PRIMARY KEY,
                    ai_chat TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    body TEXT,
                    tool_calls TEXT,
                    tool_result TEXT,
                    system_prompt TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    model TEXT,
                    provider TEXT,
                    created_date TEXT NOT NULL,
                    modified_date TEXT NOT NULL,
                    FOREIGN KEY(ai_chat) REFERENCES ai_chat(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_aichat ON ai_chat_message(ai_chat);")

            # 3. FTS5 Virtual Table for full-text search
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS ai_chat_message_fts USING fts5(
                        ai_chat UNINDEXED,
                        message_id UNINDEXED,
                        role UNINDEXED,
                        body,
                        tokenize='porter unicode61'
                    );
                    """
                )
            except sqlite3.OperationalError:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS ai_chat_message_fts USING fts5(
                        ai_chat UNINDEXED,
                        message_id UNINDEXED,
                        role UNINDEXED,
                        body
                    );
                    """
                )

    def save_session(self, session: SessionState, title: Optional[str] = None):
        """Atomically saves or updates ai_chat and persists child ai_chat_message records."""
        now_str = datetime.now(timezone.utc).isoformat()
        created_str = (
            session.created_at.isoformat()
            if isinstance(session.created_at, datetime)
            else str(session.created_at)
        )

        chat_name = title or session.name
        if not chat_name:
            for m in session.messages:
                if m.role == "user" and (m.body or m.content):
                    clean = (m.body or m.content or "").strip().replace("\n", " ")
                    chat_name = clean[:50] + ("..." if len(clean) > 50 else "")
                    break
        if not chat_name:
            chat_name = f"Chat {session.session_id}"
        session.name = chat_name

        with self._get_connection() as conn:
            # 1. Upsert ai_chat record
            conn.execute(
                """
                INSERT INTO ai_chat (
                    id, name, status, created_date, modified_date, provider, model,
                    total_input_tokens, total_output_tokens, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    modified_date = excluded.modified_date,
                    provider = excluded.provider,
                    model = excluded.model,
                    total_input_tokens = excluded.total_input_tokens,
                    total_output_tokens = excluded.total_output_tokens,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    session.session_id,
                    chat_name,
                    session.status,
                    created_str,
                    now_str,
                    session.active_provider,
                    session.active_model,
                    session.total_input_tokens,
                    session.total_output_tokens,
                    json.dumps(session.metadata),
                ),
            )

            # 2. Sync child ai_chat_message records
            existing_msg_ids = {
                row["id"]
                for row in conn.execute("SELECT id FROM ai_chat_message WHERE ai_chat = ?", (session.session_id,))
            }

            # If messages were evicted (e.g. during context compaction), purge them from SQLite
            current_msg_ids = {m.id for m in session.messages if m.id}
            if current_msg_ids:
                evicted_ids = existing_msg_ids - current_msg_ids
                if evicted_ids:
                    evicted_list = list(evicted_ids)
                    chunk_size = 400
                    for i in range(0, len(evicted_list), chunk_size):
                        chunk = evicted_list[i : i + chunk_size]
                        placeholders = ",".join("?" for _ in chunk)
                        conn.execute(
                            f"DELETE FROM ai_chat_message WHERE ai_chat = ? AND id IN ({placeholders})",
                            [session.session_id, *chunk],
                        )
                        try:
                            conn.execute(
                                f"DELETE FROM ai_chat_message_fts WHERE ai_chat = ? AND message_id IN ({placeholders})",
                                [session.session_id, *chunk],
                            )
                        except Exception:
                            pass

            for msg in session.messages:
                msg_body = msg.body or msg.content or ""
                tool_calls_raw = (
                    [tc.model_dump() for tc in msg.tool_calls]
                    if msg.tool_calls
                    else None
                )
                tool_result_raw = (
                    msg.tool_result.model_dump()
                    if msg.tool_result
                    else None
                )
                input_tok = msg.usage.prompt_tokens if msg.usage else 0
                output_tok = msg.usage.completion_tokens if msg.usage else 0
                created_ts_str = (
                    msg.created_date.isoformat()
                    if isinstance(msg.created_date, datetime)
                    else str(msg.created_date)
                )
                modified_ts_str = (
                    msg.modified_date.isoformat()
                    if isinstance(msg.modified_date, datetime)
                    else str(msg.modified_date)
                )

                if msg.id not in existing_msg_ids:
                    conn.execute(
                        """
                        INSERT INTO ai_chat_message (
                            id, ai_chat, name, role, body, tool_calls, tool_result,
                            system_prompt, input_tokens, output_tokens, model, provider,
                            created_date, modified_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            msg.id,
                            session.session_id,
                            f"Message {msg.id}",
                            msg.role,
                            msg_body,
                            _json_dumps(tool_calls_raw) if tool_calls_raw else None,
                            _json_dumps(tool_result_raw) if tool_result_raw else None,
                            msg.system_prompt,
                            input_tok,
                            output_tok,
                            msg.model or session.active_model,
                            msg.provider or session.active_provider,
                            created_ts_str,
                            modified_ts_str,
                        ),
                    )

                    # Index text into FTS5 if non-empty
                    if msg_body.strip():
                        try:
                            conn.execute(
                                """
                                INSERT INTO ai_chat_message_fts (ai_chat, message_id, role, body)
                                VALUES (?, ?, ?, ?)
                                """,
                                (session.session_id, msg.id, msg.role, msg_body.strip()),
                            )
                        except Exception:
                            pass
                else:
                    # Update body and tool_result if modified by compaction
                    conn.execute(
                        """
                        UPDATE ai_chat_message
                        SET body = ?, tool_result = ?, modified_date = ?
                        WHERE id = ? AND ai_chat = ?
                        """,
                        (
                            msg_body,
                            _json_dumps(tool_result_raw) if tool_result_raw else None,
                            now_str,
                            msg.id,
                            session.session_id,
                        ),
                    )

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Reconstructs full SessionState from ai_chat and ai_chat_message tables."""
        with self._get_connection() as conn:
            chat_row = conn.execute("SELECT * FROM ai_chat WHERE id = ?", (session_id,)).fetchone()
            if not chat_row:
                return None

            metadata = json.loads(chat_row["metadata_json"] or "{}")

            msg_rows = conn.execute(
                "SELECT * FROM ai_chat_message WHERE ai_chat = ? ORDER BY created_date ASC",
                (session_id,),
            ).fetchall()

            messages: List[Message] = []
            for r in msg_rows:
                tool_calls = None
                if r["tool_calls"]:
                    tc_list = json.loads(r["tool_calls"])
                    tc_list = _restore_bytes_recursive(tc_list)
                    tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc.get("arguments", {}),
                            thought_signature=tc.get("thought_signature"),
                        )
                        for tc in tc_list
                    ]

                tool_result = None
                if r["tool_result"]:
                    tr = json.loads(r["tool_result"])
                    tool_result = ToolResult(
                        call_id=tr["call_id"],
                        name=tr["name"],
                        output=tr["output"],
                        is_error=tr.get("is_error", False),
                        duration_ms=tr.get("duration_ms", 0.0),
                        payload_bytes=tr.get("payload_bytes", 0),
                        item_count=tr.get("item_count"),
                        truncated=tr.get("truncated", False),
                    )

                usage = UsageMetadata(
                    prompt_tokens=r["input_tokens"] or 0,
                    completion_tokens=r["output_tokens"] or 0,
                    total_tokens=(r["input_tokens"] or 0) + (r["output_tokens"] or 0),
                )

                ts = datetime.fromisoformat(r["created_date"]) if r["created_date"] else datetime.now(timezone.utc)
                mod_ts = datetime.fromisoformat(r["modified_date"]) if r["modified_date"] else ts

                messages.append(
                    Message(
                        id=r["id"],
                        ai_chat=r["ai_chat"],
                        role=r["role"],
                        body=r["body"],
                        content=r["body"],
                        model=r["model"],
                        provider=r["provider"],
                        system_prompt=r["system_prompt"],
                        tool_calls=tool_calls,
                        tool_result=tool_result,
                        usage=usage,
                        created_date=ts,
                        modified_date=mod_ts,
                    )
                )

            created_dt = (
                datetime.fromisoformat(chat_row["created_date"])
                if chat_row["created_date"]
                else datetime.now(timezone.utc)
            )
            modified_dt = (
                datetime.fromisoformat(chat_row["modified_date"])
                if chat_row["modified_date"]
                else datetime.now(timezone.utc)
            )

            session = SessionState(
                session_id=chat_row["id"],
                name=chat_row["name"],
                status=chat_row["status"],
                active_model=chat_row["model"],
                active_provider=chat_row["provider"],
                messages=messages,
                created_at=created_dt,
                modified_at=modified_dt,
                metadata=metadata,
                total_input_tokens=chat_row["total_input_tokens"] or 0,
                total_output_tokens=chat_row["total_output_tokens"] or 0,
            )
            return session

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent ai_chat sessions with message counts and tokens."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT 
                    c.id, c.name, c.status, c.created_date, c.modified_date, c.provider, c.model,
                    c.total_input_tokens, c.total_output_tokens,
                    (c.total_input_tokens + c.total_output_tokens) as total_tokens,
                    COUNT(m.id) as message_count
                FROM ai_chat c
                LEFT JOIN ai_chat_message m ON c.id = m.ai_chat
                GROUP BY c.id
                ORDER BY c.modified_date DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {
                    "session_id": r["id"],
                    "title": r["name"],
                    "status": r["status"],
                    "created_at": r["created_date"],
                    "updated_at": r["modified_date"],
                    "active_model": r["model"],
                    "active_provider": r["provider"],
                    "total_input_tokens": r["total_input_tokens"],
                    "total_output_tokens": r["total_output_tokens"],
                    "total_tokens": r["total_tokens"],
                    "message_count": r["message_count"],
                }
                for r in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        """Deletes an ai_chat session and all its child messages."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM ai_chat WHERE id = ?", (session_id,))
            conn.execute("DELETE FROM ai_chat_message_fts WHERE ai_chat = ?", (session_id,))
            return cur.rowcount > 0

    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search across past message turns using FTS5."""
        clean_q = query.strip()
        if not clean_q:
            return []

        fts_query = " ".join(f'"{token}"' for token in clean_q.split() if token)

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT 
                    f.ai_chat as session_id, f.message_id, f.role, f.body,
                    c.name as title, c.modified_date as updated_at, c.model as active_model,
                    snippet(ai_chat_message_fts, 3, '<b>', '</b>', '...', 15) as snippet_text
                FROM ai_chat_message_fts f
                JOIN ai_chat c ON f.ai_chat = c.id
                WHERE ai_chat_message_fts MATCH ?
                ORDER BY bm25(ai_chat_message_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

            return [
                {
                    "session_id": r["session_id"],
                    "message_id": r["message_id"],
                    "role": r["role"],
                    "title": r["title"],
                    "snippet": r["snippet_text"] or r["body"][:120],
                    "updated_at": r["updated_at"],
                    "active_model": r["active_model"],
                }
                for r in rows
            ]
