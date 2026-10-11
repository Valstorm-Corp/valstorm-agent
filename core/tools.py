"""Tool definitions, @tool decorator, and ToolRegistry for agent runtime."""

import ast
import inspect
import json
import math
import operator
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin
import uuid

from core.models import ToolResult
from core.sanitizer import sanitize_text


# Type mapping from Python types to JSON Schema types
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _parse_docstring(docstring: Optional[str]) -> tuple[str, Dict[str, str]]:
    """Extract general description and per-argument descriptions from docstring."""
    if not docstring:
        return "", {}
    
    lines = inspect.cleandoc(docstring).splitlines()
    desc_lines: List[str] = []
    param_docs: Dict[str, str] = {}
    
    in_args_section = False
    current_param: Optional[str] = None
    current_param_desc: List[str] = []
    
    for line in lines:
        stripped = line.strip()
        
        # Detect Args: or Parameters: section
        if re.match(r"^(Args|Arguments|Parameters):", stripped, re.IGNORECASE):
            in_args_section = True
            continue
        elif re.match(r"^(Returns|Raises|Example|Examples|Note|Notes):", stripped, re.IGNORECASE):
            in_args_section = False
            if current_param:
                param_docs[current_param] = " ".join(current_param_desc).strip()
                current_param = None
                current_param_desc = []
            continue
            
        # Sphinx style :param x: description
        sphinx_match = re.match(r"^:param\s+(\w+):\s*(.*)", stripped)
        if sphinx_match:
            p_name, p_desc = sphinx_match.groups()
            param_docs[p_name] = p_desc.strip()
            continue
            
        if in_args_section:
            # Google style: param_name (type): description or param_name: description
            param_match = re.match(r"^(\w+)(?:\s*\([^)]+\))?:\s*(.*)", stripped)
            if param_match:
                if current_param:
                    param_docs[current_param] = " ".join(current_param_desc).strip()
                current_param, p_desc = param_match.groups()
                current_param_desc = [p_desc.strip()] if p_desc else []
            elif current_param and (line.startswith("    ") or line.startswith("\t")):
                current_param_desc.append(stripped)
        else:
            desc_lines.append(line)
            
    if current_param:
        param_docs[current_param] = " ".join(current_param_desc).strip()
        
    description = "\n".join(desc_lines).strip()
    return description, param_docs


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema property dict."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    
    # Handle Optional[T] / Union[T, None]
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # Filter out type(None)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        return {"type": "string"}
        
    if origin in (list, List):
        args = get_args(annotation)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}
        
    if origin in (dict, Dict):
        return {"type": "object"}
        
    if annotation in TYPE_MAP:
        return {"type": TYPE_MAP[annotation]}
        
    return {"type": "string"}


def tool(
    func_or_name: Optional[Union[Callable, str]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Decorator to mark a function as an agent tool and generate its JSON Schema.
    
    Can be used as:
        @tool
        def my_func(a: int) -> str: ...
        
        @tool(name="custom_name", description="custom desc")
        def my_func(a: int) -> str: ...
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or (func_or_name if isinstance(func_or_name, str) else fn.__name__)
        doc_desc, param_docs = _parse_docstring(fn.__doc__)
        tool_desc = description or doc_desc or f"Tool: {tool_name}"
        
        # Introspect signature
        sig = inspect.signature(fn)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        
        for p_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
                
            prop_schema = _python_type_to_json_schema(param.annotation)
            if p_name in param_docs:
                prop_schema["description"] = param_docs[p_name]
            properties[p_name] = prop_schema
            
            # If no default value, it is required
            if param.default is inspect.Parameter.empty:
                required.append(p_name)
                
        schema = {
            "name": tool_name,
            "description": tool_desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        
        # Attach tool metadata to function
        setattr(fn, "_is_tool", True)
        setattr(fn, "_tool_name", tool_name)
        setattr(fn, "_tool_description", tool_desc)
        setattr(fn, "_tool_schema", schema)
        return fn

    if callable(func_or_name):
        return decorator(func_or_name)
    return decorator


def _save_tool_artifact(call_id: str, raw_str: str) -> Optional[str]:
    """Saves raw tool output to ~/.valstorm/artifacts/<call_id>.log for complete traceability."""
    try:
        art_dir = Path.home() / ".valstorm" / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        art_file = art_dir / f"{call_id}.log"
        art_file.write_text(raw_str, encoding="utf-8")
        return str(art_file)
    except Exception:
        return None


def _sanitize_and_window_output(
    res: Any, max_bytes: int = 16_000, call_id: Optional[str] = None
) -> tuple[str, int, Optional[int], bool, Optional[int]]:
    """Sanitizes tool output and windows/truncates oversized payloads to prevent context blowup.

    For oversized payloads (> max_bytes):
      - Saves complete raw output to ~/.valstorm/artifacts/<call_id>.log
      - Returns head and tail lines with a clear artifact reference marker.

    Returns:
        (output_str, payload_bytes, item_count, is_truncated, raw_size_bytes)
    """
    item_count = None
    if isinstance(res, (list, tuple)):
        item_count = len(res)
    elif isinstance(res, dict):
        item_count = len(res.keys())

    if isinstance(res, (dict, list, tuple)):
        try:
            raw_str = json.dumps(res, indent=2, ensure_ascii=False)
        except Exception:
            raw_str = str(res)
    else:
        raw_str = str(res)

    # Sanitize API keys, auth tokens, database credentials and sensitive environment variables
    raw_str = sanitize_text(raw_str)

    raw_bytes = len(raw_str.encode("utf-8"))

    if raw_bytes > max_bytes:
        cid = call_id or str(uuid.uuid4())
        art_path = _save_tool_artifact(cid, raw_str)
        
        # Split into head and tail chunks
        half_bytes = max_bytes // 2
        head = raw_str[:half_bytes]
        tail = raw_str[-half_bytes:]
        
        art_ref = f" Full output saved to: {art_path}" if art_path else ""
        notice = (
            f"\n\n[... Payload Windowed: Showing head ({len(head.encode('utf-8'))} bytes) "
            f"and tail ({len(tail.encode('utf-8'))} bytes) of {raw_bytes} total bytes.{art_ref} ...]\n\n"
        )
        final_str = head + notice + tail
        return final_str, len(final_str.encode("utf-8")), item_count, True, raw_bytes

    return raw_str, raw_bytes, item_count, False, raw_bytes


class ToolRegistry:
    """Registry managing tool definitions, schema extraction, and safe execution."""

    def __init__(self, default_timeout_sec: float = 120.0, max_payload_bytes: int = 16_000) -> None:
        self._tools: Dict[str, Callable] = {}
        self.default_timeout_sec = default_timeout_sec
        self.max_payload_bytes = max_payload_bytes

    def register(
        self,
        func_or_tool: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """Register a function or @tool decorated callable into the registry.
        
        Can be used as a method or decorator:
            registry.register(func)
            @registry.register
            @registry.register(name="custom")
        """
        def decorator(fn: Callable) -> Callable:
            if not getattr(fn, "_is_tool", False):
                decorated = tool(name=name, description=description)(fn)
            else:
                decorated = fn
                if name:
                    decorated._tool_name = name
                    decorated._tool_schema["name"] = name
                if description:
                    decorated._tool_description = description
                    decorated._tool_schema["description"] = description
                    
            tool_name = getattr(decorated, "_tool_name", fn.__name__)
            self._tools[tool_name] = decorated
            return decorated

        if callable(func_or_tool):
            return decorator(func_or_tool)
        return decorator

    def get(self, tool_name: str) -> Optional[Callable]:
        """Retrieve a registered tool function by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List names of all registered tools."""
        return list(self._tools.keys())

    def filter_by_whitelist(self, allowed_tools: List[str]) -> "ToolRegistry":
        """Return a new ToolRegistry instance containing only the tools in allowed_tools."""
        scoped = ToolRegistry(
            default_timeout_sec=self.default_timeout_sec,
            max_payload_bytes=self.max_payload_bytes,
        )
        allowed_set = set(allowed_tools)
        for name, fn in self._tools.items():
            if name in allowed_set:
                scoped.register(fn)
        return scoped

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return standard JSON schemas for all registered tools."""
        schemas: List[Dict[str, Any]] = []
        for tool_func in self._tools.values():
            if hasattr(tool_func, "_tool_schema"):
                schemas.append(tool_func._tool_schema)
            else:
                decorated = tool(tool_func)
                schemas.append(decorated._tool_schema)
        return schemas

    def execute(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> ToolResult:
        """Safely execute a registered tool with precision timing and payload windowing."""
        import time
        c_id = call_id or str(uuid.uuid4())
        args = args or {}
        start_t = time.perf_counter()

        tool_func = self._tools.get(tool_name)
        if not tool_func:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error: Tool '{tool_name}' is not registered.",
                is_error=True,
                duration_ms=dur_ms,
            )

        try:
            if inspect.iscoroutinefunction(tool_func):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            res = executor.submit(asyncio.run, tool_func(**args)).result()
                    else:
                        res = loop.run_until_complete(tool_func(**args))
                except RuntimeError:
                    res = asyncio.run(tool_func(**args))
            else:
                res = tool_func(**args)

            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            out_str, p_bytes, i_count, is_trunc, raw_size = _sanitize_and_window_output(
                res, max_bytes=self.max_payload_bytes, call_id=c_id
            )

            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=out_str,
                is_error=False,
                duration_ms=dur_ms,
                payload_bytes=p_bytes,
                item_count=i_count,
                truncated=is_trunc,
                raw_size_bytes=raw_size,
            )
        except TypeError as te:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error calling '{tool_name}' with arguments {args}: {te}",
                is_error=True,
                duration_ms=dur_ms,
            )
        except Exception as e:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error executing tool '{tool_name}': {e}",
                is_error=True,
                duration_ms=dur_ms,
            )

    async def execute_async(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> ToolResult:
        """Asynchronously execute a registered tool with precision timing, timeout guardrails, and windowing."""
        import asyncio
        import time

        c_id = call_id or str(uuid.uuid4())
        args = args or {}
        timeout = timeout_sec or self.default_timeout_sec
        start_t = time.perf_counter()

        tool_func = self._tools.get(tool_name)
        if not tool_func:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error: Tool '{tool_name}' is not registered.",
                is_error=True,
                duration_ms=dur_ms,
            )

        try:
            if inspect.iscoroutinefunction(tool_func):
                res = await asyncio.wait_for(tool_func(**args), timeout=timeout)
            else:
                res = tool_func(**args)

            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            out_str, p_bytes, i_count, is_trunc, raw_size = _sanitize_and_window_output(
                res, max_bytes=self.max_payload_bytes, call_id=c_id
            )

            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=out_str,
                is_error=False,
                duration_ms=dur_ms,
                payload_bytes=p_bytes,
                item_count=i_count,
                truncated=is_trunc,
                raw_size_bytes=raw_size,
            )
        except asyncio.TimeoutError:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error: Tool execution timed out after {timeout:.1f}s",
                is_error=True,
                duration_ms=dur_ms,
            )
        except TypeError as te:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error calling '{tool_name}' with arguments {args}: {te}",
                is_error=True,
                duration_ms=dur_ms,
            )
        except Exception as e:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                call_id=c_id,
                name=tool_name,
                output=f"Error executing tool '{tool_name}': {e}",
                is_error=True,
                duration_ms=dur_ms,
            )


# ==========================================
# Built-in Safe Math Calculator
# ==========================================

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.pow,  # Common alias for exponentiation in user queries
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST expression node with safety whitelisting."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable or function: '{node.id}'")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        func = _eval_ast_node(node.func)
        if not callable(func):
            raise ValueError(f"Object '{node.func}' is not callable")
        args = [_eval_ast_node(arg) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    else:
        raise ValueError(f"Unsupported syntax expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate basic mathematical expressions.
    
    Args:
        expression: The mathematical expression to evaluate (e.g. '(45 * 12) + 180', 'sqrt(144)', '2 ** 8').
        
    Returns:
        The evaluated result as a string, or an error message.
    """
    clean_expr = expression.strip()
    if not clean_expr:
        return "Error: Empty expression provided."
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_ast_node(parsed)
        # Format integer result cleanly if float is exact int
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def read_local_file(path: str) -> str:
    """Read the contents of a local text file.
    
    Args:
        path: Path to the local file to read.
        
    Returns:
        The content of the file or an error message.
    """
    try:
        resolved_path = Path(os.path.expanduser(path)).resolve()
        if not resolved_path.exists():
            return f"Error: File not found at '{path}'."
        if not resolved_path.is_file():
            return f"Error: '{path}' is not a file."
        return resolved_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file '{path}': {e}"


@tool
def mock_db_lookup(query: str) -> str:
    """Return mock database JSON records for testing queries.
    
    Args:
        query: Search query or table identifier (e.g. 'users', 'orders', 'products', or a keyword).
        
    Returns:
        JSON string representation of matched records.
    """
    q = query.lower()
    
    if "user" in q or "customer" in q:
        records = [
            {"id": "usr_101", "name": "Alice Johnson", "email": "alice@example.com", "role": "admin", "status": "active"},
            {"id": "usr_102", "name": "Bob Smith", "email": "bob@example.com", "role": "member", "status": "active"},
            {"id": "usr_103", "name": "Charlie Brown", "email": "charlie@example.com", "role": "viewer", "status": "suspended"},
        ]
    elif "order" in q or "purchase" in q:
        records = [
            {"id": "ord_501", "customer_id": "usr_101", "total_amount": 149.99, "currency": "USD", "status": "fulfilled"},
            {"id": "ord_502", "customer_id": "usr_102", "total_amount": 34.50, "currency": "USD", "status": "pending"},
            {"id": "ord_503", "customer_id": "usr_101", "total_amount": 89.00, "currency": "USD", "status": "processing"},
        ]
    elif "product" in q or "item" in q or "inventory" in q:
        records = [
            {"id": "prod_001", "title": "Wireless Noise-Canceling Headphones", "price": 199.99, "stock": 45, "category": "Electronics"},
            {"id": "prod_002", "title": "Ergonomic Mechanical Keyboard", "price": 129.50, "stock": 18, "category": "Electronics"},
            {"id": "prod_003", "title": "Ultra-Wide Monitor Stand", "price": 49.99, "stock": 120, "category": "Accessories"},
        ]
    else:
        records = [
            {"query": query, "matches_found": 1, "record": {"id": "rec_001", "title": f"Record matching '{query}'", "data": "Sample payload data", "timestamp": "2026-08-17T00:00:00Z"}},
        ]
        
    return json.dumps(records, indent=2)


def get_default_registry() -> ToolRegistry:
    """Return a ToolRegistry pre-loaded with built-in tools."""
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(read_local_file)
    registry.register(mock_db_lookup)
    return registry
