"""Pure Python asyncio RESP Redis Client with zero external dependencies.

Supports standard Redis operations (PING, AUTH, GET, SET, INCR, EXPIRE, RPOP, LPUSH)
over raw asyncio TCP streams. Works seamlessly in minimal environments where redis-py is not installed.
"""

import asyncio
import logging
from typing import Any, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger("vsagent.resp_redis")


class AsyncRespRedisClient:
    """Zero-dependency asyncio Redis client using RESP2 protocol."""

    def __init__(self, url: str = "redis://localhost:6379/0"):
        self.url = url
        parsed = urlparse(url)
        self.scheme = parsed.scheme or "redis"
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 6379
        self.username = parsed.username
        self.password = parsed.password
        self.db = int(parsed.path.lstrip("/") or "0") if parsed.path else 0
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Establishes connection and handles AUTH / SELECT if configured."""
        if self.writer and not self.writer.is_closing():
            return

        ssl_ctx = None
        if self.scheme == "rediss":
            import ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        self.reader, self.writer = await asyncio.open_connection(self.host, self.port, ssl=ssl_ctx)
        if self.username and self.password:
            await self.execute("AUTH", self.username, self.password)
        elif self.password:
            await self.execute("AUTH", self.password)
        if self.db != 0:
            await self.execute("SELECT", str(self.db))

    async def close(self):
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None

    async def _read_resp(self) -> Any:
        line = await self.reader.readline()
        if not line:
            raise ConnectionResetError("Redis connection closed by peer")

        prefix = line[:1]
        content = line[1:-2]  # strip trailing \r\n

        if prefix == b"+":
            # Simple string
            return content.decode("utf-8", errors="replace")
        elif prefix == b"-":
            # Error
            err_msg = content.decode("utf-8", errors="replace")
            raise RuntimeError(f"Redis Error: {err_msg}")
        elif prefix == b":":
            # Integer
            return int(content)
        elif prefix == b"$":
            # Bulk string
            length = int(content)
            if length == -1:
                return None
            data = await self.reader.readexactly(length + 2)  # includes \r\n
            return data[:-2].decode("utf-8", errors="replace")
        elif prefix == b"*":
            # Array
            count = int(content)
            if count == -1:
                return None
            items = []
            for _ in range(count):
                item = await self._read_resp()
                items.append(item)
            return items
        else:
            raise ValueError(f"Unknown RESP byte: {prefix}")

    async def execute(self, command: str, *args: Union[str, int, bytes]) -> Any:
        """Executes a Redis command and parses the RESP response."""
        async with self._lock:
            await self.connect()
            cmd_args = [command] + list(args)
            msg = f"*{len(cmd_args)}\r\n"
            for arg in cmd_args:
                val = str(arg) if not isinstance(arg, bytes) else arg.decode("utf-8", errors="replace")
                val_bytes = val.encode("utf-8")
                msg += f"${len(val_bytes)}\r\n{val}\r\n"

            self.writer.write(msg.encode("utf-8"))
            await self.writer.drain()
            return await self._read_resp()

    async def get(self, key: str) -> Optional[str]:
        return await self.execute("GET", key)

    async def set(
        self,
        key: str,
        value: Union[str, int],
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        cmd = ["SET", key, str(value)]
        if ex is not None:
            cmd.extend(["EX", str(ex)])
        if nx:
            cmd.append("NX")
        res = await self.execute(*cmd)
        return res == "OK"

    async def incr(self, key: str) -> int:
        return await self.execute("INCR", key)

    async def expire(self, key: str, seconds: int) -> bool:
        res = await self.execute("EXPIRE", key, str(seconds))
        return res == 1

    async def rpop(self, key: str) -> Optional[str]:
        return await self.execute("RPOP", key)

    async def hget(self, key: str, field: str) -> Optional[str]:
        return await self.execute("HGET", key, field)

    async def lpush(self, key: str, *values: str) -> int:
        return await self.execute("LPUSH", key, *values)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
