# Valstorm REST API Tools Integration (Internal Developers)

## Security & Architecture Principle
The Agent Runtime strictly calls Valstorm's FastAPI backend over authenticated REST endpoints with HTTP/2 connection pooling. **Zero direct database connections (MongoDB/Qdrant/Redis) are permitted from client code.**

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Runtime Client                     │
│  • valstorm_sql_query          • valstorm_vfs_search        │
│  • valstorm_record_cud         • valstorm_vfs_browse        │
│  • valstorm_schema_inspect                                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP/2 Connection Pool
                               │ Authorization: Bearer ***
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Valstorm FastAPI Backend                    │
│  • Depends(get_current_user) -> Resolves tenant org DB      │
│  • RBAC & field-level read/write permissions                │
│  • Formulas & Rollup summary evaluation                     │
│  • Audit logging & CUD triggers                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tool Endpoint Mapping

| Runtime Tool | FastAPI Endpoint | Purpose |
| :--- | :--- | :--- |
| `valstorm_sql_query` | `POST /query` | SQL query engine with custom keywords (`ME`, `today`, `PHONE:`, `JOIN`). |
| `valstorm_vfs_search` | `POST /v1/search` | Hybrid vector (Qdrant) + metadata keyword search with RRF blending. |
| `valstorm_record_cud` | `POST/PATCH/DELETE /object/{api_name}` | Batch record creation, update, and deletion. |
| `valstorm_vfs_browse` | `GET /vfs/vault/{id}`, `GET /vfs/tree` | Vault folder traversal and file listing. |
| `valstorm_schema_inspect` | `GET /schema/{api_name}` | On-demand schema discovery (fields, types, validation rules). |

---

## Authentication & Profile Resolution

Credentials and active workspaces are automatically resolved in hierarchical order:

1. **Explicit Override**: `--valstorm-token <token>` argument.
2. **Workspace & CLI Auth Profile Discovery**:
   - Searches upward for `valstorm.json` (e.g. `{"env": "local", "profile": "vdk"}`).
   - Resolves `~/.valstorm/auth_{env}_{profile}.json` (e.g. `~/.valstorm/auth_local_vdk.json`).
   - Fallbacks to `~/.valstorm/auth_{env}.json` or `~/.valstorm/auth.json`.
3. **KeyStore & Environment Variables**:
   - Checks `VALSTORM_API_TOKEN`, `VALSTORM_JWT_TOKEN`, `VALSTORM_PAT`.
   - Reads workspace `.env`, `.env.ai.keys`, or `~/.config/valstorm/keys.json`.

---

## Transparent Token Auto-Healing

When an `access_token` expires or signature verification fails with `401 Unauthorized`:
1. `ValstormApiClient` intercepts the `401` response.
2. Calls `POST /oauth2/refresh` using the profile's `refresh_token`.
3. Saves the newly issued `access_token` and `refresh_token` back to disk (`~/.valstorm/auth_{env}_{profile}.json`).
4. Re-injects the fresh `Authorization: Bearer <new_token>` header into the active HTTP/2 client session.
5. Retries the in-flight request transparently without failing the agent run.

---

## Audit & Debug Logging

Every credential resolution step, file inspection, auto-refresh attempt, and endpoint request is appended with UTC timestamps to:
```text
apps/agent-runtime/auth_debug.log
```
