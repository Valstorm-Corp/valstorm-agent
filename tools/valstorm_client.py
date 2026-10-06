"""Valstorm Async REST API Client for Agent Runtime.

Provides fast, authenticated HTTP/2 access to Valstorm platform endpoints:
- SQL Query Engine (/query)
- Virtual File Service (/vfs & /v1/search)
- Schema Metadata Discovery (/schema)
- Batch Record Mutations (/object/{api_name})
- Transparent Auto-Refresh on 401 Unauthorized via /oauth2/refresh
"""

import contextvars
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from core.keystore import KeyStore

_current_valstorm_auth: contextvars.ContextVar[Optional[Tuple[str, str]]] = contextvars.ContextVar(
    "_current_valstorm_auth", default=None
)


def set_current_valstorm_auth(token: Optional[str], base_url: Optional[str]) -> contextvars.Token:
    """Sets the active scoped Valstorm credentials for the current async task/turn."""
    return _current_valstorm_auth.set((token, base_url) if (token and base_url) else None)


def get_current_valstorm_auth() -> Optional[Tuple[str, str]]:
    """Gets the active scoped Valstorm credentials for the current async task/turn."""
    return _current_valstorm_auth.get()

ENV_URL_MAP = {
    "local": "http://localhost:8010/v1",
    "dev": "https://api-dev.valstorm.com/v1",
    "prod": "https://api.valstorm.com/v1",
    "blue": "http://localhost:8011/v1",
    "green": "http://localhost:8021/v1",
}

AUTH_LOG_FILE = Path(__file__).parent.parent / "auth_debug.log"


def log_auth_debug(message: str):
    """Appends timestamped authentication debug logs to auth_debug.log."""
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with open(AUTH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{now_str}] {message}\n")
    except Exception:
        pass


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decodes JWT payload without cryptographic verification."""
    import base64
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload_b64 = parts[1]
        padding = len(payload_b64) % 4
        if padding:
            payload_b64 += "=" * (4 - padding)
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return {}


def is_jwt_expired(token: str, margin_seconds: int = 60) -> bool:
    """Checks whether token is an expired or nearly-expired JWT."""
    import time
    payload = decode_jwt_payload(token)
    if not payload or "exp" not in payload:
        return False
    try:
        exp = float(payload["exp"])
        return time.time() >= (exp - margin_seconds)
    except (ValueError, TypeError):
        return False


def get_refresh_endpoint(base_url: str) -> str:
    """Computes correct OAuth2 refresh endpoint URL from base_url."""
    clean = base_url.rstrip("/")
    if clean.endswith("/v1"):
        return f"{clean}/oauth2/refresh"
    return f"{clean}/v1/oauth2/refresh"


def save_tokens_to_auth_file(
    auth_file_path: Path,
    access_token: str,
    refresh_token: Optional[str] = None,
) -> bool:
    """Atomically persists refreshed tokens into ~/.valstorm auth profile JSON file or desktop tokens.json."""
    try:
        auth_file_path = Path(auth_file_path)
        creds: Dict[str, Any] = {}
        if auth_file_path.is_file():
            try:
                with open(auth_file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        creds = content
            except Exception:
                creds = {}
        
        is_desktop_file = "com.valstorm.app" in str(auth_file_path)
        if is_desktop_file:
            creds["accessToken"] = access_token
            if refresh_token:
                creds["refreshToken"] = refresh_token
        else:
            creds["access_token"] = access_token
            if refresh_token:
                creds["refresh_token"] = refresh_token

        auth_file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = auth_file_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(creds, f, indent=2)
        temp_file.replace(auth_file_path)
        log_auth_debug(f"Saved refreshed tokens back to {auth_file_path}")

        # Also sync to default ~/.valstorm CLI profile if desktop tokens were updated
        try:
            valstorm_cli_default = Path.home() / ".valstorm" / "auth_prod_default.json"
            if is_desktop_file and valstorm_cli_default.exists():
                c_data = {}
                try:
                    c_data = json.loads(valstorm_cli_default.read_text())
                except Exception:
                    pass
                c_data["access_token"] = access_token
                if refresh_token:
                    c_data["refresh_token"] = refresh_token
                valstorm_cli_default.write_text(json.dumps(c_data, indent=2))
        except Exception:
            pass

        return True
    except Exception as e:
        log_auth_debug(f"Failed saving tokens to {auth_file_path}: {e}")
        return False


def refresh_valstorm_tokens_sync(
    base_url: str,
    refresh_token: str,
    auth_file_path: Optional[Path] = None,
    timeout: float = 10.0,
) -> Optional[Tuple[str, str]]:
    """Synchronously refreshes an access token using POST /oauth2/refresh."""
    if not refresh_token:
        return None
    endpoint = get_refresh_endpoint(base_url)
    log_auth_debug(f"Executing sync token refresh via {endpoint}...")
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json={"refresh_token": refresh_token})
            if resp.status_code == 200:
                data = resp.json()
                new_access = data.get("access_token")
                new_refresh = data.get("refresh_token") or refresh_token
                if new_access:
                    if auth_file_path:
                        save_tokens_to_auth_file(auth_file_path, new_access, new_refresh)
                    log_auth_debug(f"Sync token refresh succeeded! New token prefix: {new_access[:10]}...")
                    return new_access, new_refresh
            else:
                log_auth_debug(f"Refresh failed (sync) {resp.status_code}: {resp.text}")
    except Exception as e:
        log_auth_debug(f"Network error during sync token refresh: {e}")
    return None


async def refresh_valstorm_tokens_async(
    base_url: str,
    refresh_token: str,
    auth_file_path: Optional[Path] = None,
    timeout: float = 10.0,
) -> Optional[Tuple[str, str]]:
    """Asynchronously refreshes an access token using POST /oauth2/refresh."""
    if not refresh_token:
        return None
    endpoint = get_refresh_endpoint(base_url)
    log_auth_debug(f"Executing async token refresh via {endpoint}...")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json={"refresh_token": refresh_token})
            if resp.status_code == 200:
                data = resp.json()
                new_access = data.get("access_token")
                new_refresh = data.get("refresh_token") or refresh_token
                if new_access:
                    if auth_file_path:
                        save_tokens_to_auth_file(auth_file_path, new_access, new_refresh)
                    log_auth_debug(f"Async token refresh succeeded! New token prefix: {new_access[:10]}...")
                    return new_access, new_refresh
            else:
                log_auth_debug(f"Refresh failed (async) {resp.status_code}: {resp.text}")
    except Exception as e:
        log_auth_debug(f"Network error during async token refresh: {e}")
    return None


def _find_workspace_valstorm_config() -> dict:
    """Helper to locate and read valstorm.json by searching parent directories."""
    curr = Path.cwd().resolve()
    for directory in [curr, *curr.parents]:
        config_file = directory / "valstorm.json"
        if config_file.is_file():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                log_auth_debug(f"Error reading valstorm.json at {config_file}: {e}")
    return {}


def resolve_valstorm_auth_context(
    override_token: Optional[str] = None,
    env: Optional[str] = None,
    override_base_url: Optional[str] = None,
    profile: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Optional[Path]]:
    """Resolves Valstorm API token, Base URL, Refresh Token, and active Auth File."""
    # Check ContextVar for active async run/turn scoped credentials
    scoped_auth = get_current_valstorm_auth()
    if scoped_auth and not override_token and not override_base_url:
        s_token, s_base_url = scoped_auth
        if s_token:
            clean_url = (s_base_url or "http://localhost:8010/v1").rstrip("/")
            if not clean_url.endswith("/v1"):
                clean_url = f"{clean_url}/v1"
            return s_token, clean_url, None, None

    ws_config = _find_workspace_valstorm_config()
    detected_env = (
        (env.strip() if env and env.strip() else None)
        or os.environ.get("VALSTORM_ENV")
        or ws_config.get("env")
        or "prod"
    ).lower().strip()
    detected_profile = (
        (profile.strip() if profile and profile.strip() else None)
        or os.environ.get("VALSTORM_PROFILE")
        or ws_config.get("profile")
        or "default"
    ).lower().strip()

    # 1. Base URL Resolution
    if override_base_url:
        base_url = override_base_url.rstrip("/")
        if not base_url.endswith("/v1") and "/v1" not in base_url:
            base_url = f"{base_url}/v1"
    elif os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL"):
        env_base = (os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL", "")).rstrip("/")
        if not env_base.endswith("/v1") and "/v1" not in env_base:
            env_base = f"{env_base}/v1"
        base_url = env_base
    else:
        base_url = ENV_URL_MAP.get(
            detected_env,
            f"http://localhost:8010/v1" if detected_env == "local" else f"https://api-{detected_env}.valstorm.com/v1",
        )

    log_auth_debug(
        f"--- Resolving Valstorm Credentials (env={detected_env}, profile={detected_profile}, base_url={base_url}) ---"
    )

    token = None
    refresh_token = None
    auth_file_path: Optional[Path] = None
    token_source = None

    # Step 1: Explicit Override Token
    if override_token and override_token.strip():
        token = override_token.strip()
        token_source = "explicit_override_argument"
        log_auth_debug(f"Found token from explicit override argument (length={len(token)})")

    # Step 2: Valstorm CLI Auth Profiles (~/.valstorm/auth_{env}_{profile}.json)
    if not token:
        valstorm_dir = Path.home() / ".valstorm"
        candidate_cli_files = [
            valstorm_dir / f"auth_{detected_env}_{detected_profile}.json",
            valstorm_dir / f"auth_{detected_env}_default.json",
            valstorm_dir / f"auth_{detected_env}.json",
            valstorm_dir / "auth_prod_default.json",
            valstorm_dir / "auth_prod.json",
            valstorm_dir / "auth.json",
        ]
        if valstorm_dir.exists():
            for extra_path in sorted(valstorm_dir.glob("auth_*.json")):
                if extra_path not in candidate_cli_files:
                    candidate_cli_files.append(extra_path)

        # Fallback to local desktop app storage (macOS, Linux, Windows)
        desktop_token_candidates = [
            Path.home() / "Library/Application Support/com.valstorm.app/tokens.json",
            Path.home() / "Library/Application Support/com.valstorm.app/tokens_dev.json",
            Path.home() / ".config/com.valstorm.app/tokens.json",
            Path.home() / ".config/com.valstorm.app/tokens_dev.json",
        ]
        app_data_env = os.environ.get("APPDATA")
        if app_data_env:
            desktop_token_candidates.extend([
                Path(app_data_env) / "com.valstorm.app/tokens.json",
                Path(app_data_env) / "com.valstorm.app/tokens_dev.json",
            ])
        for dt_path in desktop_token_candidates:
            if dt_path not in candidate_cli_files:
                candidate_cli_files.append(dt_path)

        for cli_path in candidate_cli_files:
            log_auth_debug(f"Checking CLI auth file: {cli_path} (exists={cli_path.is_file()})")
            if cli_path.is_file():
                try:
                    with open(cli_path, "r", encoding="utf-8") as f:
                        creds = json.load(f)
                        if isinstance(creds, dict):
                            t = (
                                creds.get("access_token")
                                or creds.get("accessToken")
                                or creds.get("token")
                                or creds.get("jwt")
                                or creds.get("pat")
                            )
                            if t and str(t).strip():
                                candidate_tok = str(t).strip()
                                candidate_ref = creds.get("refresh_token") or creds.get("refreshToken")

                                # If token is expired and has no refresh token to heal it, skip to next candidate file
                                if is_jwt_expired(candidate_tok) and not candidate_ref:
                                    log_auth_debug(
                                        f"Token in {cli_path} is expired and has no refresh_token. Skipping..."
                                    )
                                    continue

                                token = candidate_tok
                                refresh_token = candidate_ref
                                auth_file_path = cli_path
                                token_source = str(cli_path)
                                # If env was not explicitly passed, infer base_url from auth file name
                                if not env and not override_base_url:
                                    parts = cli_path.stem.split("_")
                                    if len(parts) >= 2 and parts[1] in ENV_URL_MAP:
                                        base_url = ENV_URL_MAP[parts[1]]

                                # Proactively refresh expired or near-expiry JWT if refresh_token is present
                                if refresh_token and is_jwt_expired(token):
                                    log_auth_debug(
                                        f"Token in {cli_path} is expired (or near expiry). Proactively refreshing via {base_url}..."
                                    )
                                    refreshed = refresh_valstorm_tokens_sync(
                                        base_url=base_url,
                                        refresh_token=refresh_token,
                                        auth_file_path=cli_path,
                                    )
                                    if refreshed:
                                        token, refresh_token = refreshed
                                        log_auth_debug(
                                            f"Proactive refresh succeeded! New access token prefix={token[:10]}..."
                                        )
                                    else:
                                        log_auth_debug(
                                            "Proactive refresh failed; proceeding with existing token."
                                        )

                                log_auth_debug(
                                    f"Successfully loaded token from {cli_path} (length={len(token)}, has_refresh={bool(refresh_token)})"
                                )
                                break
                except Exception as err:
                    log_auth_debug(f"Failed parsing {cli_path}: {err}")

    # Step 3: Direct Environment Variables & .env Files via KeyStore
    if not token:
        log_auth_debug("Checking KeyStore (environment variables, .env, .env.ai.keys)...")
        token = KeyStore.resolve_key("valstorm")
        if token:
            token_source = "KeyStore (.env / environment variable)"
            log_auth_debug(f"Found token via KeyStore (length={len(token)})")

    if not token:
        log_auth_debug("WARNING: No Valstorm authentication token could be resolved from any source.")
    else:
        log_auth_debug(
            f"FINAL AUTH RESOLUTION: Source={token_source}, Base URL={base_url}, Token Prefix={token[:10]}..."
        )

    return token or "", base_url, refresh_token, auth_file_path


def resolve_valstorm_credentials(
    override_token: Optional[str] = None,
    env: Optional[str] = None,
    override_base_url: Optional[str] = None,
    profile: Optional[str] = None,
) -> Tuple[str, str]:
    """Convenience 2-tuple resolver returning (access_token, base_url)."""
    token, base_url, _, _ = resolve_valstorm_auth_context(
        override_token=override_token,
        env=env,
        override_base_url=override_base_url,
        profile=profile,
    )
    return token, base_url


class ValstormApiClient:
    """High-performance async HTTP client for Valstorm platform REST API with auto-refresh."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        env: Optional[str] = None,
        timeout: float = 20.0,
        client: Optional[httpx.AsyncClient] = None,
        refresh_token: Optional[str] = None,
        auth_file_path: Optional[Union[str, Path]] = None,
    ):
        resolved_token, resolved_base_url, refresh_tok, auth_file = resolve_valstorm_auth_context(
            override_token=token,
            env=env,
            override_base_url=base_url,
        )
        self.token = resolved_token
        self.base_url = resolved_base_url.rstrip("/") + "/"
        self.refresh_token = refresh_token or refresh_tok
        self.auth_file_path = auth_file_path or auth_file
        self.timeout = timeout

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "ValstormAgentRuntime/1.0",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
            self._owns_client = True

    async def _try_refresh_token(self) -> bool:
        """Attempts to refresh an expired access token using refresh_token."""
        if not self.refresh_token:
            log_auth_debug("Cannot auto-refresh token: No refresh_token available.")
            return False

        log_auth_debug("Attempting auto-refresh of access token via POST /oauth2/refresh...")
        tokens = await refresh_valstorm_tokens_async(
            base_url=self.base_url,
            refresh_token=self.refresh_token,
            auth_file_path=self.auth_file_path,
        )
        if tokens:
            self.token, self.refresh_token = tokens
            self._client.headers["Authorization"] = f"Bearer {self.token}"
            log_auth_debug(f"Auto-refresh succeeded! New access token prefix={self.token[:10]}...")
            return True
        return False

    async def _attempt_refresh(self, context: str):
        """Internal helper to execute the token refresh logic."""
        log_auth_debug(f"Executing {context} token refresh inside lock.")
        refreshed = await refresh_valstorm_tokens_async(
            base_url=self.base_url,
            refresh_token=self.refresh_token,
            auth_file_path=self.auth_file_path,
        )
        if refreshed:
            self.token = refreshed[0]
            self.refresh_token = refreshed[1]
            self._client.headers["Authorization"] = f"Bearer {self.token}"
        else:
            self.token = "" # Invalidate the access token to ensure no further attempts
            self.refresh_token = "" # Invalidate the refresh token
            log_auth_debug(f"Failed to refresh Valstorm token during {context}. Refresh token invalid. Cleared tokens.")
            return False

    async def _request_with_retry(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Sends HTTP request and automatically retries with refreshed token on 401."""
        self._ensure_authenticated()
        clean_endpoint = endpoint.lstrip("/")

        # 1. Proactive Token Refresh Check (before sending the request)
        if self.refresh_token and is_jwt_expired(self.token, margin_seconds=60):
            log_auth_debug("Proactive refresh triggered. Token expires in <60s.")
            async with self._lock:
                # Re-check inside lock in case another request already refreshed it
                # If still expired/near-expired, refresh it.
                if is_jwt_expired(self.token, margin_seconds=60):
                    await self._try_refresh_token()
        
        # 2. Execute Request
        resp = await self._client.request(method, clean_endpoint, **kwargs)

        if resp.status_code == 401:
            log_auth_debug(f"Received 401 on {method} {clean_endpoint}. Triggering auto-refresh flow...")
            refreshed = await self._try_refresh_token()
            if refreshed:
                if "headers" in kwargs:
                    kwargs["headers"]["Authorization"] = f"Bearer {self.token}"
                resp = await self._client.request(method, clean_endpoint, **kwargs)

        resp.raise_for_status()
        return resp

    async def close(self):
        """Closes the underlying HTTP client session."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _ensure_authenticated(self):
        if not self.token:
            err_msg = (
                "No Valstorm authentication token found. "
                "Log in via `valstorm login` or set VALSTORM_API_TOKEN in .env."
            )
            log_auth_debug(f"API Call Blocked: {err_msg}")
            raise PermissionError(err_msg)

    # ==========================================
    # 1. SQL Query Engine (/query)
    # ==========================================

    async def sql_query(self, query: str, bypass_cache: bool = False) -> Dict[str, Any]:
        """Executes a SQL query against the Valstorm query engine."""
        payload = {"query": query, "bypass_cache": bypass_cache}
        log_auth_debug(f"Sending POST /query: {query}")
        resp = await self._request_with_retry("POST", "/query", json=payload)
        data = resp.json()
        return {
            "records": data if isinstance(data, list) else data.get("records", []),
            "headers": dict(resp.headers),
        }

    async def mongo_query(self, collection: str, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes a direct MongoDB aggregation pipeline."""
        payload = {"collection": collection, "pipeline": pipeline}
        log_auth_debug(f"Sending POST /query/mongo on {collection}")
        resp = await self._request_with_retry("POST", "/query/mongo", json=payload)
        data = resp.json()
        return {
            "records": data if isinstance(data, list) else data.get("records", []),
            "headers": dict(resp.headers),
        }

    # ==========================================
    # 2. Virtual File Service (/vfs & /v1/search)
    # ==========================================

    async def vfs_get_file(self, file_id: str) -> Dict[str, Any]:
        """Loads file metadata and inline text content or presigned URL in a single roundtrip."""
        endpoint = f"/vfs/file/{file_id}"
        resp = await self._request_with_retry("GET", endpoint)
        return resp.json()

    async def vfs_get_snapshot(self) -> Dict[str, Any]:
        """Fetches full organization VFS hierarchy snapshot (vaults and files)."""
        endpoint = "/vfs/snapshot"
        resp = await self._request_with_retry("GET", endpoint)
        return resp.json()

    async def vfs_delete_item(self, item_id: str) -> Dict[str, Any]:
        """Deletes a file or vault from VFS."""
        endpoint = f"/vfs/{item_id}"
        resp = await self._request_with_retry("DELETE", endpoint)
        return resp.json()

    async def vfs_search(
        self,
        query: str,
        limit: int = 10,
        enable_rag: Optional[bool] = None,
        vault_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Executes hybrid vector + metadata search across workspace files."""
        payload: Dict[str, Any] = {"query": query, "limit": limit}
        if enable_rag is not None:
            payload["enable_rag"] = enable_rag
        if vault_ids:
            payload["vault_ids"] = vault_ids

        endpoint = "/search" if self.base_url.endswith("/v1") else "/v1/search"
        log_auth_debug(f"Sending POST {endpoint}: {payload}")
        resp = await self._request_with_retry("POST", endpoint, json=payload)
        return resp.json()

    async def vfs_browse_vault(self, vault_id: str = "root", bypass_cache: bool = False) -> Dict[str, Any]:
        """Fetches contents (child vaults and files) of a specific Vault folder."""
        endpoint = f"/vfs/vault/{vault_id}"
        resp = await self._request_with_retry("GET", endpoint, params={"bypass_cache": bypass_cache})
        return resp.json()

    async def vfs_browse_path(self, string_path: str) -> Dict[str, Any]:
        """Resolves a human path (e.g. 'Finance/2026') to vault contents."""
        clean_path = string_path.strip("/")
        endpoint = f"/vfs/path/{clean_path}"
        resp = await self._request_with_retry("GET", endpoint)
        return resp.json()

    async def vfs_get_tree(self, bypass_cache: bool = False) -> Dict[str, Any]:
        """Fetches full vault directory tree accessible to the user."""
        endpoint = "/vfs/tree"
        resp = await self._request_with_retry("GET", endpoint, params={"bypass_cache": bypass_cache})
        return resp.json()

    async def vfs_move_item(
        self,
        item_id: str,
        from_vault_id: Optional[str] = None,
        to_vault_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Moves a file or vault folder to another vault."""
        payload = {
            "item_id": item_id,
            "from_vault_id": from_vault_id,
            "to_vault_id": to_vault_id,
        }
        endpoint = "/vfs/move"
        resp = await self._request_with_retry("POST", endpoint, json=payload)
        return resp.json()

    async def vfs_write_file(
        self,
        name: str,
        content: str,
        vault_id: Optional[str] = None,
        file_id: Optional[str] = None,
        is_public: bool = False,
    ) -> Dict[str, Any]:
        """Creates or updates a document in VFS directly with its text/markdown content."""
        if file_id:
            clean_id = file_id.replace("cloud://", "")
            endpoint = f"/files/{clean_id}/content"
            resp = await self._request_with_retry("POST", endpoint, json={"content": content})
            return resp.json()
        else:
            clean_vault_id = vault_id.replace("cloud://", "") if vault_id else None
            record_payload = {
                "name": name,
                "content": content,
                "is_public": is_public,
            }
            if clean_vault_id:
                record_payload["vaults"] = [clean_vault_id]
            res = await self.records_create("file", [record_payload])
            return res

    # ==========================================
    # 3. Batch Record Mutations (/object/{api_name})
    # ==========================================

    async def records_create(self, api_name: str, records: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """Creates one or multiple records in a collection."""
        payload = records if isinstance(records, list) else [records]
        endpoint = f"/object/{api_name}"
        resp = await self._request_with_retry("POST", endpoint, json=payload)
        return resp.json()

    async def records_update(self, api_name: str, records: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """Updates one or multiple records in a collection (must include 'id')."""
        payload = records if isinstance(records, list) else [records]
        endpoint = f"/object/{api_name}"
        resp = await self._request_with_retry("PATCH", endpoint, json=payload)
        return resp.json()

    async def records_delete(self, api_name: str, ids: List[str]) -> Dict[str, Any]:
        """Deletes records by ID from a collection."""
        endpoint = f"/object/{api_name}"
        resp = await self._request_with_retry("DELETE", endpoint, params={"ids": ids})
        return {"status": "success", "deleted_count": len(ids)}

    async def records_hydrate_batch(self, ids: List[str]) -> Dict[str, Any]:
        """Resolves a batch of prefixed IDs (e.g. ['cont_123', 'task_456']) into names and schemas."""
        payload = {"ids": ids}
        resp = await self._request_with_retry("POST", "/object/hydrate-batch", json=payload)
        return resp.json()

    async def records_merge(
        self,
        master_id: str,
        duplicate_ids: Union[str, List[str]],
        schema_api_name: Optional[str] = None,
        field_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Merges one or more duplicate records into master record and re-links related lookup references."""
        selected = [duplicate_ids] if isinstance(duplicate_ids, str) else list(duplicate_ids)
        payload = {
            "master_record": master_id,
            "master_id": master_id,
            "selected_records": selected,
            "duplicate_ids": selected,
            "schema_api_name": schema_api_name or "",
            "field_overrides": field_overrides or {},
        }
        resp = await self._request_with_retry("POST", "/merge", json=payload)
        return resp.json()

    # ==========================================
    # 4. Schema Discovery & Metadata (/schema)
    # ==========================================

    async def schema_get(self, api_name: Optional[str] = None) -> Any:
        """Fetches all schemas or a specific object schema."""
        if api_name:
            endpoint = f"/schema/{api_name}"
        else:
            endpoint = "/schema"
        resp = await self._request_with_retry("GET", endpoint)
        return resp.json()

    async def schema_create(self, data: Dict[str, Any]) -> Any:
        """Creates a new custom object schema."""
        resp = await self._request_with_retry("POST", "/schema", json=data)
        return resp.json()

    async def schema_update(self, data: Dict[str, Any]) -> Any:
        """Updates an existing object schema."""
        resp = await self._request_with_retry("PATCH", "/schema", json=data)
        return resp.json()

    async def schema_create_field(self, data: Dict[str, Any]) -> Any:
        """Creates a field in an object schema."""
        resp = await self._request_with_retry("POST", "/schema/field", json=data)
        return resp.json()

    async def schema_delete(self, schema_id: str) -> Any:
        """Deletes a custom object schema."""
        resp = await self._request_with_retry("DELETE", f"/schema/{schema_id}")
        return resp.json()

    # ==========================================
    # 4.5 Automation, Dynamic Functions & Logs
    # ==========================================

    async def function_list(self) -> List[Dict[str, Any]]:
        """Queries callable custom functions stored in the organization."""
        res = await self.sql_query("SELECT id, name, file_name, description FROM function")
        return res.get("records", [])

    async def function_call(
        self,
        function_name: Optional[str] = None,
        function_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes a custom or system function on the backend platform."""
        payload: Dict[str, Any] = {"inputs": inputs or {}}
        if function_name:
            payload["function_name"] = function_name
        if function_id:
            payload["function_id"] = function_id
        resp = await self._request_with_retry("POST", "/automation/function", json=payload)
        return resp.json()

    async def function_validate(
        self,
        code: str,
        function_name: Optional[str] = "test_function",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dry-runs and validates custom Python function code through AST security checks."""
        payload = {
            "code": code,
            "function_name": function_name or "test_function",
            "inputs": inputs or {},
        }
        resp = await self._request_with_retry("POST", "/automation/function/validate", json=payload)
        return resp.json()

    async def get_execution_logs(
        self,
        target_id: Optional[str] = None,
        status: Optional[str] = None,
        log_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Queries recent execution log entries."""
        clauses = []
        if target_id:
            clauses.append(f"target_id = '{target_id}'")
        if status:
            clauses.append(f"status = '{status}'")
        if log_type:
            clauses.append(f"type = '{log_type}'")

        where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT id, name, type, target_id, target_name, status, duration_ms, error_message, created_date FROM log{where_clause} ORDER BY created_date DESC LIMIT {limit}"
        res = await self.sql_query(query=query)
        return res.get("records", [])

    async def get_log_detail(self, log_id: str) -> Dict[str, Any]:
        """Fetches decompressed inputs, outputs, and stack traces for a log record."""
        clean_id = log_id.strip()
        resp = await self._request_with_retry("GET", f"/records/log/{clean_id}/details")
        return resp.json()

    # ==========================================
    # 4. Analytics, Reports & List Filters
    # ==========================================

    async def analytics_compute(
        self,
        query: str,
        chart_type: str = "Bar",
        field: Optional[List[Dict[str, Any]]] = None,
        x: Optional[List[Dict[str, Any]]] = None,
        y: Optional[List[Dict[str, Any]]] = None,
        aggregate: str = "count",
        group_by_date_time: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> Dict[str, Any]:
        """Calls /analytics/report to compute server-side pandas statistical aggregations."""
        payload: Dict[str, Any] = {
            "query": query,
            "chartType": chart_type,
            "aggregate": aggregate,
            "bypass_cache": bypass_cache,
        }
        if field:
            payload["field"] = field
        if x:
            payload["x"] = x
        if y:
            payload["y"] = y
        if group_by_date_time:
            payload["groupByDateTime"] = group_by_date_time

        resp = await self._request_with_retry("POST", "/analytics/report", json=payload)
        return resp.json()

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
        """Creates and persists a list_filter record for DataGrid and Kanban views."""
        schema_info = await self.schema_get(object_api_name)
        schema_id = schema_info.get("id") or object_api_name
        
        # Auto-resolve display fields if none provided
        resolved_display_fields = display_fields
        if not resolved_display_fields:
            props = schema_info.get("properties", {})
            resolved_display_fields = [
                k for k in ["name", "status", "stage", "amount", "loan_amount", "email", "phone", "owner", "created_date"]
                if k in props
            ]
            if not resolved_display_fields:
                resolved_display_fields = list(props.keys())[:6]

        record_payload = {
            "name": name,
            "object": schema_id,
            "query": sql_query,
            "display_fields": resolved_display_fields,
            "pinned": is_pinned,
            "is_default": is_default,
        }
        if app_id:
            record_payload["app"] = app_id

        return await self.records_create("list_filter", [record_payload])

    async def generate_report(
        self,
        name: str,
        object_api_name: str,
        sql_query: str,
        chart_type: str = "Bar",
        x_field: Optional[str] = None,
        y_field: Optional[str] = None,
        aggregation: str = "Count",
        group_by_time: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates and persists a report record with chart_settings and query configuration."""
        schema_info = await self.schema_get(object_api_name)
        schema_id = schema_info.get("id") or object_api_name

        chart_settings: Dict[str, Any] = {
            "chartType": chart_type,
            "groupByDateTime": group_by_time,
        }
        if x_field:
            chart_settings["x"] = [{"key": x_field, "label": x_field.replace("_", " ").title()}]
        if y_field:
            chart_settings["y"] = [{
                "key": y_field,
                "label": y_field.replace("_", " ").title(),
                "aggregate": aggregation
            }]
        elif chart_type.lower() in ("donut", "pie", "funnel") and x_field:
            chart_settings["field"] = [{
                "key": x_field,
                "label": x_field.replace("_", " ").title(),
                "aggregate": aggregation
            }]

        report_payload = {
            "name": name,
            "object": schema_id,
            "query": {
                "sql_query": sql_query,
                "query_mode": "SQL",
            },
            "chart_settings": chart_settings,
            "standard": False,
        }
        if app_id:
            report_payload["app"] = app_id

        return await self.records_create("report", [report_payload])

    # ==========================================
    # 5. Scheduling, Cron Loops & Drip Cadences
    # ==========================================

    async def create_scheduled_task(
        self,
        name: str,
        target_type: str,
        target_id: str,
        run_at_utc: Optional[str] = None,
        cron_expression: Optional[str] = None,
        payload_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates either a one-off scheduled_item or a recurring schedule_trigger_setting."""
        if cron_expression:
            # Recurring cron loop
            record_payload = {
                "name": name,
                "cron_schedule": cron_expression.strip(),
                "active": True,
                "data": payload_data or {},
            }
            if target_type.lower() == "function":
                record_payload["function"] = target_id
            else:
                record_payload["automation"] = target_id
            return await self.records_create("schedule_trigger_setting", [record_payload])
        else:
            # One-off future task
            record_payload = {
                "name": name,
                "run_date_time": run_at_utc or datetime.now(timezone.utc).isoformat(),
                "status": "Queued",
                "data": payload_data or {},
            }
            if target_type.lower() == "function":
                record_payload["function"] = target_id
            else:
                record_payload["automation"] = target_id
            return await self.records_create("scheduled_item", [record_payload])

    async def create_drip_cadence(
        self,
        name: str,
        target_schema: str,
        steps: List[Dict[str, Any]],
        exit_conditions: Optional[Dict[str, Any]] = None,
        target_type: str = "automation",
        target_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates and persists a drip_definition blueprint."""
        schema_info = await self.schema_get(target_schema)
        schema_id = schema_info.get("id") or target_schema

        record_payload: Dict[str, Any] = {
            "name": name,
            "target_schema": schema_id,
            "step_configuration": steps,
            "exit_condition": exit_conditions or {},
            "active": True,
        }
        if target_id:
            if target_type.lower() == "function":
                record_payload["function"] = target_id
            else:
                record_payload["automation"] = target_id
        if app_id:
            record_payload["app"] = app_id

        return await self.records_create("drip_definition", [record_payload])

    async def enroll_in_drip(
        self,
        drip_definition_id: str,
        target_record_id: str,
        target_schema: str,
        start_date_time_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enrolls a record into an active drip campaign."""
        record_payload = {
            "name": f"Drip Enrollment: {target_record_id}",
            "drip_definition": drip_definition_id,
            "target_record": {
                "id": target_record_id,
                "schema": target_schema,
            },
            "status": "Active",
            "current_step": 1,
            "start_date_time": start_date_time_utc or datetime.now(timezone.utc).isoformat(),
        }
        return await self.records_create("drip_enrollment", [record_payload])

    # ==========================================
    # 5b. Telephony & SMS Communications (/twilio/*)
    # ==========================================

    async def send_sms(
        self,
        message: str,
        to_phone: Optional[str] = None,
        contact_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        from_number: Optional[str] = None,
        file_id: Optional[str] = None,
        is_group: bool = False,
    ) -> Dict[str, Any]:
        """Sends an SMS or MMS message via the unified TwilioContext with self-healing conversations."""
        payload = {
            "message": message,
            "to_phone": to_phone,
            "contact_id": contact_id,
            "conversation_id": conversation_id,
            "from_number": from_number,
            "file_id": file_id,
            "is_group": is_group,
        }
        resp = await self._request_with_retry("POST", "/twilio/sms/send", json=payload)
        return resp.json()

    # ==========================================
    # 6. Slack Integration (/slack/*)
    # ==========================================

    async def slack_post_message(
        self,
        channel: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        as_user: bool = False,
        mrkdwn: bool = True,
    ) -> Dict[str, Any]:
        """Posts a message or Block Kit payload to a Slack channel or thread via chat.postMessage."""
        payload: Dict[str, Any] = {
            "channel": channel,
            "as_user": as_user,
            "mrkdwn": mrkdwn,
        }
        if text:
            payload["text"] = text
        if blocks:
            payload["blocks"] = blocks
        if thread_ts:
            payload["thread_ts"] = thread_ts
        resp = await self._request_with_retry("POST", "/slack/chat/post", json=payload)
        return resp.json()

    async def slack_update_message(
        self,
        channel: str,
        ts: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Updates an existing Slack message via chat.update."""
        payload: Dict[str, Any] = {"channel": channel, "ts": ts}
        if text:
            payload["text"] = text
        if blocks:
            payload["blocks"] = blocks
        resp = await self._request_with_retry("POST", "/slack/chat/update", json=payload)
        return resp.json()

    async def slack_delete_message(self, channel: str, ts: str, as_user: bool = False) -> Dict[str, Any]:
        """Deletes a Slack message via chat.delete."""
        payload = {"channel": channel, "ts": ts, "as_user": as_user}
        resp = await self._request_with_retry("POST", "/slack/chat/delete", json=payload)
        return resp.json()

    async def slack_list_channels(
        self,
        types: str = "public_channel,private_channel",
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Lists channels in the connected Slack workspace via conversations.list."""
        params: Dict[str, Any] = {"types": types, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = await self._request_with_retry("GET", "/slack/channels", params=params)
        return resp.json()

    async def slack_get_channel_history(
        self,
        channel: str,
        limit: int = 50,
        latest: Optional[str] = None,
        oldest: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetches recent conversation history (or a thread) from a channel via conversations.history."""
        params: Dict[str, Any] = {"channel": channel, "limit": limit}
        if latest:
            params["latest"] = latest
        if oldest:
            params["oldest"] = oldest
        if cursor:
            params["cursor"] = cursor
        resp = await self._request_with_retry("GET", "/slack/chat/history", params=params)
        return resp.json()

    async def slack_list_users(self, cursor: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Lists members of the connected Slack workspace via users.list."""
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = await self._request_with_retry("GET", "/slack/users", params=params)
        return resp.json()

    async def slack_get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieves a specific Slack user's profile via users.profile.get."""
        resp = await self._request_with_retry("GET", f"/slack/users/{user_id}/profile")
        return resp.json()

    async def slack_add_reaction(self, channel: str, timestamp: str, name: str) -> Dict[str, Any]:
        """Adds an emoji reaction to a Slack message via reactions.add."""
        payload = {"channel": channel, "timestamp": timestamp, "name": name}
        resp = await self._request_with_retry("POST", "/slack/reactions/add", json=payload)
        return resp.json()

    async def slack_get_auth_status(self) -> Dict[str, Any]:
        """Returns current Slack integration connection status and workspace metadata."""
        resp = await self._request_with_retry("GET", "/slack/auth/status")
        return resp.json()
