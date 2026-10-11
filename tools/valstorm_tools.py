"""High-precision Valstorm Platform tools for Agent Runtime."""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from core.tools import ToolRegistry, tool
from tools.valstorm_client import ValstormApiClient
from tools.slack_tools import create_slack_tools


def create_valstorm_tools(client: Optional[ValstormApiClient] = None):
    """Creates tool functions bound to a specific ValstormApiClient instance."""
    api_client = client or ValstormApiClient()

    @tool
    async def valstorm_sql_query(query: str, limit: int = 50, bypass_cache: bool = False) -> str:
        """Executes a SQL query against the Valstorm query engine.

        Supports standard SQL keywords, dynamic date filters (e.g. today, this_month, last_n_days:7),
        JOIN operations, and the ME keyword for user-scoped filtering.

        Args:
            query: The SQL query string (e.g. "SELECT id, name, email FROM contact WHERE status = 'Active' LIMIT 10").
            limit: Maximum number of records to return (default: 50).
            bypass_cache: Set true to bypass Redis query cache and hit primary database directly.
        """
        # Ensure LIMIT is present if not specified
        clean_q = query.strip()
        if "limit" not in clean_q.lower() and limit:
            clean_q += f" LIMIT {limit}"

        res = await api_client.sql_query(query=clean_q, bypass_cache=bypass_cache)
        records = res.get("records", [])
        if not records:
            return "Query executed successfully: 0 records found."

        return json.dumps(records, indent=2, default=str)

    @tool
    async def valstorm_vfs_search(query: str, limit: int = 10, enable_rag: bool = False) -> str:
        """Executes hybrid dense vector + metadata search across documents and files in the Virtual File Service.

        Args:
            query: Natural language query, question, or filename (e.g. "quarterly financial report" or "invoice_2024.pdf").
            limit: Number of top blended hits to return (default: 10).
            enable_rag: Whether to trigger RAG synthesis and return cited excerpts.
        """
        res = await api_client.vfs_search(query=query, limit=limit, enable_rag=enable_rag)
        results = res.get("results", [])
        if not results:
            return f"No matching files or document chunks found for query: '{query}'"

        formatted_hits = []
        for r in results[:limit]:
            hit = {
                "file_id": r.get("file_id"),
                "filename": r.get("filename"),
                "score": r.get("rrf_score") or r.get("score"),
                "sources": r.get("sources"),
                "snippet": r.get("snippet") or (r.get("chunks", [{}])[0].get("text", "") if r.get("chunks") else ""),
            }
            formatted_hits.append(hit)

        return json.dumps(formatted_hits, indent=2, default=str)

    @tool
    async def valstorm_vfs_discover_knowledge(
        tags: Optional[Union[str, List[str]]] = None,
        query: Optional[str] = None,
        limit: int = 25,
    ) -> str:
        """Discovers canonical SOPs, Playbooks, Specs, and knowledge files matching Knowledge Graph domain tags.

        Args:
            tags: Optional list of Knowledge Graph tags (e.g. ['SOP', 'Engineering']) or comma-separated string. If omitted, uses active persona tags or discovers canonical knowledge.
            query: Optional search keyword to filter matching file names or locations.
            limit: Maximum number of knowledge files to return (default: 25).
        """
        tag_list: List[str] = []
        if isinstance(tags, str):
            clean_str = tags.strip()
            if clean_str.startswith("[") and clean_str.endswith("]"):
                try:
                    tag_list = json.loads(clean_str)
                except Exception:
                    tag_list = [t.strip().strip("'\"") for t in clean_str.strip("[]").split(",") if t.strip()]
            else:
                tag_list = [t.strip().strip("'\"") for t in clean_str.split(",") if t.strip()]
        elif isinstance(tags, list):
            tag_list = [str(t).strip() for t in tags if str(t).strip()]

        if not tag_list:
            from core.context import get_current_profile
            prof = get_current_profile()
            if prof:
                prof_tags = prof.get("tag") or prof.get("tags") or []
                if isinstance(prof_tags, list):
                    tag_list = [str(t).strip() for t in prof_tags if str(t).strip()]

        where_clauses: List[str] = []
        if tag_list:
            tag_conditions = ", ".join(f"'{t}'" for t in tag_list)
            where_clauses.append(f"tag IN ({tag_conditions})")
        if query and query.strip():
            safe_q = query.strip().replace("'", "''")
            where_clauses.append(f"name LIKE '%{safe_q}%'")

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"SELECT id, name, file_size, location, link, vault_paths, tag FROM file {where_str} ORDER BY name ASC LIMIT {limit}"

        res = await api_client.sql_query(sql, bypass_cache=True)
        records = res if isinstance(res, list) else (res.get("records", []) if isinstance(res, dict) else [])

        if not records:
            scope_desc = f"with tags {tag_list}" if tag_list else "in VFS"
            return f"No knowledge documents or SOP files found {scope_desc}."

        lines = [f"### 📚 Discovered Knowledge Documents ({len(records)} found):"]
        if tag_list:
            lines.append(f"*Active Knowledge Graph Tags Filter:* {', '.join(f'`{t}`' for t in tag_list)}\n")
        lines.append("| Document Name | File ID | Tags | Size |")
        lines.append("| :--- | :--- | :--- | :--- |")

        for r in records:
            doc_name = r.get("name") or "Untitled"
            f_id = r.get("id") or "-"
            f_size = r.get("file_size")
            size_str = f"{f_size} B" if f_size else "-"
            if f_size and isinstance(f_size, (int, float)):
                if f_size >= 1024 * 1024:
                    size_str = f"{f_size / (1024 * 1024):.1f} MB"
                elif f_size >= 1024:
                    size_str = f"{f_size / 1024:.1f} KB"
            r_tags = r.get("tag") or []
            tag_str = ", ".join(r_tags) if isinstance(r_tags, list) else str(r_tags)
            lines.append(f"| **{doc_name}** | `{f_id}` | {tag_str} | {size_str} |")

        lines.append("\n*Tip: Use `valstorm_vfs_get_file(file_id='<id>')` to inspect the full contents of any document.*")
        return "\n".join(lines)

    @tool
    async def valstorm_record_cud(
        api_name: str,
        action: Literal["create", "update", "delete"],
        records: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    ) -> str:
        """Performs batch record mutations (create, update, or delete) on a collection with strict schema validation.

        Args:
            api_name: The API name of the object/schema (e.g. 'contact', 'task', 'deal', 'account').
            action: Mutation action: 'create', 'update', or 'delete'.
            records: List of record dictionaries to mutate (for update/delete, must contain 'id'). Can be a JSON string.
        """
        # Parse JSON string if provided
        data_records = records
        if isinstance(records, str):
            try:
                data_records = json.loads(records)
            except Exception as e:
                return f"Error: records argument must be valid JSON: {e}"

        if isinstance(data_records, dict):
            data_records = [data_records]

        if not isinstance(data_records, list):
            return "Error: records must be a list of dictionaries."

        if action == "create":
            res = await api_client.records_create(api_name=api_name, records=data_records)
            return json.dumps(res, indent=2, default=str)
        elif action == "update":
            res = await api_client.records_update(api_name=api_name, records=data_records)
            return json.dumps(res, indent=2, default=str)
        elif action == "delete":
            ids = [str(r.get("id")) for r in data_records if isinstance(r, dict) and r.get("id")]
            if not ids and isinstance(data_records, list) and all(isinstance(x, str) for x in data_records):
                ids = [str(x) for x in data_records]
            if not ids:
                return "Error: delete action requires a list containing 'id' fields."
            res = await api_client.records_delete(api_name=api_name, ids=ids)
            return json.dumps(res, indent=2, default=str)
        else:
            return f"Error: Unsupported action '{action}'. Choose 'create', 'update', or 'delete'."

    @tool
    async def valstorm_vfs_browse(
        target: str = "root",
        browse_type: Literal["vault", "path", "tree"] = "vault",
    ) -> str:
        """Explores Virtual File Service folders, file lists, and directory trees without downloading binary content.

        Args:
            target: The Vault ID (e.g. 'root' or 'vaul_123') or string path (e.g. 'Finance/2026') to inspect.
            browse_type: 'vault' to inspect by Vault ID, 'path' to resolve human folder path, or 'tree' for full vault hierarchy.
        """
        if browse_type == "tree":
            res = await api_client.vfs_get_tree()
            return json.dumps(res, indent=2, default=str)
        elif browse_type == "path":
            res = await api_client.vfs_browse_path(string_path=target)
            return json.dumps(res, indent=2, default=str)
        else:
            res = await api_client.vfs_browse_vault(vault_id=target)
            return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_schema_inspect(api_name: str = "") -> str:
        """Discovers object definitions, available fields, relationship lookups, and validation rules in the workspace.

        Args:
            api_name: Optional object API name to inspect (e.g. 'contact', 'task'). If omitted or empty, lists all schemas in the org.
        """
        clean_name = api_name.strip() if api_name else None
        res = await api_client.schema_get(api_name=clean_name)
        if not clean_name and isinstance(res, dict):
            # Return high-level summary of object names and titles
            summary = [
                {"api_name": k, "title": v.get("title") or v.get("name"), "prefix": v.get("prefix")}
                for k, v in res.items()
                if isinstance(v, dict)
            ]
            return json.dumps(summary, indent=2)

        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_mongo_query(
        collection: str,
        pipeline: Union[str, List[Dict[str, Any]]],
    ) -> str:
        """Executes a MongoDB aggregation pipeline directly against the tenant database.

        Args:
            collection: The collection API name (e.g. 'contact', 'task', 'deal').
            pipeline: List of MongoDB aggregation pipeline stages (e.g. [{"$match": {"status": "Active"}}, {"$limit": 10}]). Can be a JSON string.
        """
        data_pipeline = pipeline
        if isinstance(pipeline, str):
            try:
                data_pipeline = json.loads(pipeline)
            except Exception as e:
                return f"Error: pipeline argument must be valid JSON: {e}"

        if not isinstance(data_pipeline, list):
            return "Error: pipeline must be a list of stage dictionaries."

        res = await api_client.mongo_query(collection=collection, pipeline=data_pipeline)
        records = res.get("records", [])
        return json.dumps(records, indent=2, default=str)

    @tool
    async def valstorm_vfs_get_file(file_id: str) -> str:
        """Loads complete metadata, breadcrumbs, and inline text content of a Virtual File Service (VFS) document.

        Args:
            file_id: The ID of the file (e.g. 'file_123' or document ID).
        """
        clean_id = file_id.strip()
        res = await api_client.vfs_get_file(file_id=clean_id)
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_vfs_write_file(
        name: str,
        content: str,
        vault_id: str = "",
        file_id: str = "",
        is_public: bool = False,
    ) -> str:
        """Creates or updates a document or markdown file in a Virtual File Service (VFS) Knowledge Vault.

        Use this dedicated VFS tool whenever creating, writing, or updating markdown files, SOPs,
        notes, pricing sheets, specifications, or documents in Cloud Vaults.

        Args:
            name: The filename with extension (e.g. 'Valstorm_Pricing.md' or 'Q3_Strategy.md').
            content: The complete markdown or text content to store in the document.
            vault_id: Optional target Vault ID (e.g. 'vaul_123'). If omitted, saved to root vault.
            file_id: Optional existing file ID to overwrite/update content.
            is_public: Whether the document should be publicly readable without auth.
        """
        res = await api_client.vfs_write_file(
            name=name,
            content=content,
            vault_id=vault_id or None,
            file_id=file_id or None,
            is_public=is_public,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_records_hydrate_batch(ids: Union[str, List[str]]) -> str:
        """Batch resolves a list of prefixed record IDs (e.g. ['cont_123', 'task_456', 'user_789']) into their display names and schema types.

        Args:
            ids: List of prefixed IDs to resolve. Can be a comma-separated string or JSON list.
        """
        id_list = ids
        if isinstance(ids, str):
            if ids.startswith("["):
                try:
                    id_list = json.loads(ids)
                except Exception:
                    id_list = [x.strip() for x in ids.split(",") if x.strip()]
            else:
                id_list = [x.strip() for x in ids.split(",") if x.strip()]

        if not isinstance(id_list, list):
            return "Error: ids must be a list of string IDs."

        res = await api_client.records_hydrate_batch(ids=id_list)
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_record_merge(
        master_record_id: str,
        duplicate_record_ids: Union[str, List[str]],
        schema_api_name: str = "",
    ) -> str:
        """Merges one or more duplicate records into a master record and re-links all related child documents.

        CRITICAL TWO-STEP WORKFLOW PROTOCOL:
        1. FIND & SHOW (Preview Phase): Query and present the master and duplicate records to the user in a side-by-side Markdown comparison table, noting which record will be preserved and which will be deleted. Ask for explicit user confirmation BEFORE invoking this tool.
        2. EXECUTE (Merge Phase): Only execute this tool after the user confirms. All child lookups (tasks, deals, emails, calls, notes) pointing to duplicate_record_ids are automatically re-pointed to master_record_id, and duplicate records are safely deleted.

        Args:
            master_record_id: The primary record ID to keep (e.g. 'con_08s8yZFQyQ6GfMF7').
            duplicate_record_ids: A single duplicate record ID or list of duplicate record IDs to merge into master (e.g. ['con_123456789']).
            schema_api_name: Optional collection/schema API name (e.g. 'contact', 'account', 'lead'). If omitted, inferred from ID prefix.
        """
        duplicates = duplicate_record_ids
        if isinstance(duplicate_record_ids, str):
            if duplicate_record_ids.startswith("["):
                try:
                    duplicates = json.loads(duplicate_record_ids)
                except Exception:
                    duplicates = [x.strip() for x in duplicate_record_ids.split(",") if x.strip()]
            else:
                duplicates = [duplicate_record_ids.strip()]

        if not isinstance(duplicates, list) or not duplicates:
            return "Error: duplicate_record_ids must be a non-empty list of IDs or a single ID string."

        res = await api_client.records_merge(
            master_id=master_record_id,
            duplicate_ids=duplicates,
            schema_api_name=schema_api_name.strip() if schema_api_name else None,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_function_list() -> str:
        """Lists all custom and serverless Python functions deployed in the organization."""
        res = await api_client.function_list()
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_function_call(
        function_name: str = "",
        function_id: str = "",
        inputs: Union[str, Dict[str, Any]] = "",
    ) -> str:
        """Executes a custom Python function on the Valstorm platform.

        Args:
            function_name: The name of the function to execute (e.g. 'calculate_mrr').
            function_id: The ID of the function (e.g. 'fun_123').
            inputs: JSON string or dictionary of keyword inputs to pass to the function.
        """
        input_data = inputs
        if isinstance(inputs, str) and inputs.strip():
            try:
                input_data = json.loads(inputs)
            except Exception as e:
                return f"Error: inputs must be valid JSON: {e}"
        elif not inputs:
            input_data = {}

        res = await api_client.function_call(
            function_name=function_name.strip() if function_name else None,
            function_id=function_id.strip() if function_id else None,
            inputs=input_data if isinstance(input_data, dict) else {},
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_validate_function(
        code: str,
        function_name: str = "test_function",
        inputs: Union[str, Dict[str, Any]] = "",
    ) -> str:
        """Dry-runs Python code through AST security checks (gatekeeper.py) without persisting to DB.

        Args:
            code: The Python function source code to validate.
            function_name: Identifier name for the function.
            inputs: Optional JSON string or dictionary of test inputs for dry-run execution.
        """
        input_data = inputs
        if isinstance(inputs, str) and inputs.strip():
            try:
                input_data = json.loads(inputs)
            except Exception as e:
                return f"Error: inputs must be valid JSON: {e}"
        elif not inputs:
            input_data = {}

        res = await api_client.function_validate(
            code=code,
            function_name=function_name or "test_function",
            inputs=input_data if isinstance(input_data, dict) else {},
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_get_execution_logs(
        target_id: str = "",
        status: str = "",
        log_type: str = "",
        limit: int = 10,
    ) -> str:
        """Queries recent execution log headers from the unified 'log' table.

        Args:
            target_id: Filter by ID of the function, trigger, or automation (e.g. 'fun_123').
            status: Filter by outcome status ('Success', 'Error', 'Timeout', 'Cancelled').
            log_type: Filter by engine type ('function', 'record_trigger', 'automation', 'ai_turn').
            limit: Maximum number of recent log records to return (default: 10).
        """
        res = await api_client.get_execution_logs(
            target_id=target_id.strip() if target_id else None,
            status=status.strip() if status else None,
            log_type=log_type.strip() if log_type else None,
            limit=limit,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_get_log_detail(log_id: str) -> str:
        """Fetches decompressed execution payload (inputs, outputs, and stack traces) for a log record.

        Args:
            log_id: The unique ID of the log record (e.g. 'log_1234567890abcdef').
        """
        res = await api_client.get_log_detail(log_id=log_id.strip())
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_create_list_filter(
        name: str,
        object_api_name: str,
        sql_query: str,
        display_fields: Union[str, List[str], None] = None,
        is_pinned: bool = False,
        is_default: bool = False,
        app_id: Optional[str] = None,
    ) -> str:
        """Creates and saves a list_filter record for DataGrid tables, Kanban views, and filtered campaigns.

        Args:
            name: Human-readable name for the saved filter view (e.g. 'High-Value Leads (30d)').
            object_api_name: Target object schema API name (e.g. 'lead', 'contact', 'deal', 'campaign').
            sql_query: Full SQL query (e.g. "SELECT * FROM lead WHERE owner = ME AND created_date = last_30_days ORDER BY loan_amount DESC").
            display_fields: List of column field API names to show in the table. Can be a JSON list or comma-separated string.
            is_pinned: If True, pins this filter view as the user's default active tab.
            is_default: If True, marks this filter as the global team default.
            app_id: Optional App ID to scope this filter view under.
        """
        resolved_fields = display_fields
        if isinstance(display_fields, str):
            if display_fields.startswith("["):
                try:
                    resolved_fields = json.loads(display_fields)
                except Exception:
                    resolved_fields = [x.strip() for x in display_fields.split(",") if x.strip()]
            else:
                resolved_fields = [x.strip() for x in display_fields.split(",") if x.strip()]

        res = await api_client.create_list_filter(
            name=name.strip(),
            object_api_name=object_api_name.strip(),
            sql_query=sql_query.strip(),
            display_fields=resolved_fields,
            is_pinned=is_pinned,
            is_default=is_default,
            app_id=app_id.strip() if app_id else None,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_generate_report(
        name: str,
        object_api_name: str,
        sql_query: str,
        chart_type: Literal["Bar", "Horizontal Bar", "Line", "Area", "Pie", "Donut", "Funnel", "Metric"] = "Bar",
        x_field: Optional[str] = None,
        y_field: Optional[str] = None,
        aggregation: Literal["Count", "Sum", "Average", "Min", "Max"] = "Count",
        group_by_time: Optional[Literal["Minute", "Hour", "Day", "Week", "Month", "Quarter", "Year"]] = None,
        app_id: Optional[str] = None,
    ) -> str:
        """Generates and saves a visual report record (charts, KPIs, or data tables) linked to an object.

        Args:
            name: Report name (e.g. 'Quarterly Revenue by Lead Source').
            object_api_name: Target object schema API name (e.g. 'deal', 'lead', 'invoice').
            sql_query: The underlying SQL query (e.g. "SELECT lead_source, amount FROM deal WHERE stage = 'Closed Won'").
            chart_type: Visual chart type ('Bar', 'Horizontal Bar', 'Line', 'Area', 'Pie', 'Donut', 'Funnel', 'Metric').
            x_field: Field API name for the X-axis / category / grouping (e.g. 'lead_source' or 'status').
            y_field: Field API name for the Y-axis / value metric (e.g. 'amount' or 'id').
            aggregation: Aggregation function for numeric metrics ('Count', 'Sum', 'Average', 'Min', 'Max').
            group_by_time: Time-series date bucketing for Line/Area charts ('Day', 'Week', 'Month', 'Quarter', 'Year').
            app_id: Optional App ID to associate the report with.
        """
        res = await api_client.generate_report(
            name=name.strip(),
            object_api_name=object_api_name.strip(),
            sql_query=sql_query.strip(),
            chart_type=chart_type,
            x_field=x_field.strip() if x_field else None,
            y_field=y_field.strip() if y_field else None,
            aggregation=aggregation,
            group_by_time=group_by_time,
            app_id=app_id.strip() if app_id else None,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_analytics_compute(
        sql_query: str,
        chart_type: str = "Bar",
        field: Optional[str] = None,
        x_field: Optional[str] = None,
        y_field: Optional[str] = None,
        aggregate: str = "Count",
        group_by_time: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> str:
        """Executes server-side pandas statistical aggregation via /analytics/report and returns calculated data points.

        Use this tool to compute statistical rollups (averages, sums, time-series distributions) without fetching raw record sets.

        Args:
            sql_query: Base SQL query to pull records for analysis.
            chart_type: Analytical calculation mode ('Bar', 'Line', 'Pie', 'Donut', 'Funnel', 'Metric').
            field: Field for Pie/Donut breakdown (e.g. 'status').
            x_field: Category/dimension field for Bar/Line analysis.
            y_field: Value metric field for Bar/Line analysis.
            aggregate: Aggregation operation ('Count', 'Sum', 'Average', 'Min', 'Max', 'Cumulative').
            group_by_time: Time grouping for date fields ('Day', 'Week', 'Month', 'Quarter', 'Year').
            bypass_cache: Set True to bypass Redis query cache and recompute from live DB.
        """
        field_list = [{"key": field.strip(), "aggregate": aggregate.title()}] if field else None
        x_list = [{"key": x_field.strip(), "groupByDateTime": group_by_time}] if x_field else None
        y_list = [{"key": y_field.strip(), "aggregate": aggregate.title()}] if y_field else None

        res = await api_client.analytics_compute(
            query=sql_query.strip(),
            chart_type=chart_type,
            field=field_list,
            x=x_list,
            y=y_list,
            aggregate=aggregate.lower(),
            group_by_date_time=group_by_time,
            bypass_cache=bypass_cache,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_create_scheduled_task(
        name: str,
        target_type: Literal["function", "automation"],
        target_id: str,
        run_at_utc: Optional[str] = None,
        cron_expression: Optional[str] = None,
        payload_data: Union[str, Dict[str, Any], None] = None,
    ) -> str:
        """Creates either a one-off future scheduled task (scheduled_item) or a recurring cron loop (schedule_trigger_setting).

        Args:
            name: Human-readable name for the scheduled task (e.g. 'Weekly Report Digest' or 'Lead 1042 Reminder').
            target_type: Target asset type to execute ('function' or 'automation').
            target_id: The ID of the function or automation to run (e.g. 'fun_123' or 'wf_456').
            run_at_utc: For ONE-OFF tasks: ISO-8601 UTC timestamp when this should run (e.g. '2026-09-15T14:00:00Z').
            cron_expression: For RECURRING loops: Standard cron string (e.g. '0 8 * * 1' for Mondays 8am UTC).
            payload_data: Optional JSON string or dictionary of arguments/data to pass to the function/automation.
        """
        resolved_payload = payload_data
        if isinstance(payload_data, str) and payload_data.strip():
            try:
                resolved_payload = json.loads(payload_data)
            except Exception:
                pass

        res = await api_client.create_scheduled_task(
            name=name.strip(),
            target_type=target_type,
            target_id=target_id.strip(),
            run_at_utc=run_at_utc.strip() if run_at_utc else None,
            cron_expression=cron_expression.strip() if cron_expression else None,
            payload_data=resolved_payload if isinstance(resolved_payload, dict) else None,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_create_drip_cadence(
        name: str,
        target_schema: str,
        steps: Union[str, List[Dict[str, Any]]],
        exit_conditions: Union[str, Dict[str, Any], None] = None,
        target_type: Literal["automation", "function"] = "automation",
        target_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> str:
        """Creates and configures a multi-step drip campaign blueprint (drip_definition).

        Args:
            name: Name of the campaign (e.g. '14-Day Inbound Lead Nurture').
            target_schema: Target object schema API name (e.g. 'lead', 'contact', 'deal').
            steps: List of step configurations with delays in minutes (e.g. [{"step": 1, "delay_minutes": 0}, {"step": 2, "delay_minutes": 2880}]). Can be a JSON string.
            exit_conditions: JSON dictionary of exit criteria (e.g. {"status": ["Closed Won", "Meeting Booked"]}).
            target_type: Execution engine for steps ('automation' or 'function').
            target_id: ID of the visual automation flow or Python function that handles the step routing.
            app_id: Optional App ID scope.
        """
        resolved_steps = steps
        if isinstance(steps, str):
            try:
                resolved_steps = json.loads(steps)
            except Exception as e:
                return f"Error: steps argument must be valid JSON: {e}"

        resolved_exit = exit_conditions
        if isinstance(exit_conditions, str) and exit_conditions.strip():
            try:
                resolved_exit = json.loads(exit_conditions)
            except Exception:
                pass

        res = await api_client.create_drip_cadence(
            name=name.strip(),
            target_schema=target_schema.strip(),
            steps=resolved_steps if isinstance(resolved_steps, list) else [],
            exit_conditions=resolved_exit if isinstance(resolved_exit, dict) else None,
            target_type=target_type,
            target_id=target_id.strip() if target_id else None,
            app_id=app_id.strip() if app_id else None,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_enroll_in_drip(
        drip_definition_id: str,
        target_record_id: str,
        target_schema: str,
        start_date_time_utc: Optional[str] = None,
    ) -> str:
        """Enrolls a specific record into an active drip campaign sequence (drip_enrollment).

        Args:
            drip_definition_id: The ID of the Drip Definition blueprint (e.g. 'drip_123').
            target_record_id: The ID of the lead, contact, or record to enroll (e.g. 'lead_456').
            target_schema: The schema API name of the target record (e.g. 'lead').
            start_date_time_utc: Optional start date/time in UTC ISO format. If omitted, starts immediately.
        """
        res = await api_client.enroll_in_drip(
            drip_definition_id=drip_definition_id.strip(),
            target_record_id=target_record_id.strip(),
            target_schema=target_schema.strip(),
            start_date_time_utc=start_date_time_utc.strip() if start_date_time_utc else None,
        )
        return json.dumps(res, indent=2, default=str)

    @tool
    async def valstorm_send_sms(
        message: str,
        to_phone: Optional[str] = None,
        contact_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        from_number: Optional[str] = None,
        file_id: Optional[str] = None,
        is_group: bool = False,
    ) -> str:
        """Sends an SMS or MMS text message to a contact, phone number, or conversation via Twilio Conversations.

        Automatically resolves or initializes a 1-on-1 or group conversation with proper contact linkage,
        enforces E.164 phone formatting, self-heals author participation errors, and tracks message delivery in MongoDB.

        Args:
            message: Text message content to send.
            to_phone: Recipient phone number in E.164 format (e.g. '+15551234567').
            contact_id: Recipient Valstorm contact ID (e.g. 'con_12345'). Highly recommended when messaging CRM contacts.
            conversation_id: Existing Twilio Conversation ID ('twco_...' or SID 'CH...').
            from_number: Optional tenant Twilio phone number to send from. If omitted, uses default registered phone.
            file_id: Optional Cloud Knowledge Vault / VFS file ID (e.g. 'file_123') for MMS attachments.
            is_group: Whether this is a multi-party group MMS conversation.
        """
        if not message or not message.strip():
            return "Error: message text cannot be empty."
        if not to_phone and not contact_id and not conversation_id:
            return "Error: You must provide at least one of to_phone, contact_id, or conversation_id."

        res = await api_client.send_sms(
            message=message.strip(),
            to_phone=to_phone.strip() if to_phone else None,
            contact_id=contact_id.strip() if contact_id else None,
            conversation_id=conversation_id.strip() if conversation_id else None,
            from_number=from_number.strip() if from_number else None,
            file_id=file_id.strip() if file_id else None,
            is_group=is_group,
        )
        return json.dumps(res, indent=2, default=str)

    core_tools = [
        valstorm_sql_query,
        valstorm_mongo_query,
        valstorm_vfs_search,
        valstorm_vfs_discover_knowledge,
        valstorm_vfs_browse,
        valstorm_vfs_get_file,
        valstorm_vfs_write_file,
        valstorm_record_cud,
        valstorm_record_merge,
        valstorm_records_hydrate_batch,
        valstorm_schema_inspect,
        valstorm_function_list,
        valstorm_function_call,
        valstorm_validate_function,
        valstorm_get_execution_logs,
        valstorm_get_log_detail,
        valstorm_create_list_filter,
        valstorm_generate_report,
        valstorm_analytics_compute,
        valstorm_create_scheduled_task,
        valstorm_create_drip_cadence,
        valstorm_enroll_in_drip,
        valstorm_send_sms,
    ]
    slack_tools = create_slack_tools(client=api_client)
    return core_tools + slack_tools


def register_valstorm_tools(
    registry: ToolRegistry,
    client: Optional[ValstormApiClient] = None,
    token: Optional[str] = None,
    env: str = "local",
) -> ToolRegistry:
    """Registers the 5 Valstorm platform tools into a ToolRegistry."""
    api_client = client or ValstormApiClient(token=token, env=env)
    tools = create_valstorm_tools(client=api_client)
    for t in tools:
        registry.register(t)
    return registry
