"""Unit tests for tool definitions, @tool decorator, and ToolRegistry."""

import json
from pathlib import Path
import pytest
from typing import List, Optional

from core.models import ToolResult
from core.tools import (
    ToolRegistry,
    calculator,
    get_default_registry,
    mock_db_lookup,
    read_local_file,
    tool,
)


class TestToolDecoratorAndSchema:
    """Tests for @tool introspection and JSON Schema generation."""

    def test_basic_tool_decorator(self):
        @tool
        def add_numbers(a: int, b: int = 10) -> int:
            """Add two numbers together.
            
            Args:
                a: The first number.
                b: The second number to add.
            """
            return a + b

        assert getattr(add_numbers, "_is_tool") is True
        schema = getattr(add_numbers, "_tool_schema")
        assert schema["name"] == "add_numbers"
        assert "Add two numbers together" in schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert "a" in schema["parameters"]["properties"]
        assert schema["parameters"]["properties"]["a"]["type"] == "integer"
        assert schema["parameters"]["properties"]["a"]["description"] == "The first number."
        assert schema["parameters"]["properties"]["b"]["type"] == "integer"
        assert schema["parameters"]["required"] == ["a"]

    def test_custom_name_and_description(self):
        @tool(name="custom_fetch", description="Fetch custom data from API")
        def fetch_data(query: str, limit: Optional[int] = 5) -> str:
            return f"results for {query}"

        schema = getattr(fetch_data, "_tool_schema")
        assert schema["name"] == "custom_fetch"
        assert schema["description"] == "Fetch custom data from API"
        assert schema["parameters"]["properties"]["query"]["type"] == "string"
        assert schema["parameters"]["properties"]["limit"]["type"] == "integer"
        assert schema["parameters"]["required"] == ["query"]

    def test_type_mappings(self):
        @tool
        def complex_tool(
            s: str,
            i: int,
            f: float,
            b: bool,
            items: List[str],
            metadata: dict,
        ) -> str:
            return "ok"

        schema = getattr(complex_tool, "_tool_schema")
        props = schema["parameters"]["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"
        assert props["metadata"]["type"] == "object"
        assert len(schema["parameters"]["required"]) == 6


class TestToolRegistry:
    """Tests for ToolRegistry registration and execution."""

    def test_register_methods(self):
        registry = ToolRegistry()

        # Method 1: direct function
        def func_a(x: int) -> int:
            """Func A"""
            return x * 2

        registry.register(func_a)
        assert "func_a" in registry.list_tools()

        # Method 2: decorator without parens
        @registry.register
        def func_b(msg: str) -> str:
            """Func B"""
            return msg.upper()

        assert "func_b" in registry.list_tools()

        # Method 3: decorator with custom name
        @registry.register(name="custom_c", description="Custom C")
        def func_c() -> str:
            return "done"

        assert "custom_c" in registry.list_tools()
        assert len(registry.list_tools()) == 3

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(calculator)
        registry.register(mock_db_lookup)

        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "calculator" in names
        assert "mock_db_lookup" in names

    def test_execute_success(self):
        registry = ToolRegistry()

        @registry.register
        def multiply(x: int, y: int) -> int:
            return x * y

        result = registry.execute("multiply", {"x": 6, "y": 7}, call_id="call_001")
        assert isinstance(result, ToolResult)
        assert result.call_id == "call_001"
        assert result.name == "multiply"
        assert result.output == "42"
        assert result.is_error is False

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("non_existent", {"arg": "val"}, call_id="call_002")
        assert result.is_error is True
        assert "not registered" in result.output

    def test_execute_argument_error(self):
        registry = ToolRegistry()

        @registry.register
        def strict_func(required_arg: str) -> str:
            return required_arg

        result = registry.execute("strict_func", {"wrong_arg": "val"})
        assert result.is_error is True
        assert "Error calling" in result.output

    def test_execute_runtime_exception(self):
        registry = ToolRegistry()

        @registry.register
        def faulty_tool() -> str:
            raise ValueError("Something went wrong internally")

        result = registry.execute("faulty_tool", {})
        assert result.is_error is True
        assert "Something went wrong internally" in result.output

    @pytest.mark.asyncio
    async def test_execute_async(self):
        registry = ToolRegistry()

        @registry.register
        async def async_fetch(url: str) -> str:
            return f"content from {url}"

        result = await registry.execute_async("async_fetch", {"url": "https://example.com"})
        assert result.is_error is False
        assert result.output == "content from https://example.com"


class TestBuiltinTools:
    """Tests for built-in calculator, read_local_file, and mock_db_lookup tools."""

    def test_calculator_basic_operations(self):
        assert calculator("(45 * 12) + 180") == "720"
        assert calculator("10 / 4") == "2.5"
        assert calculator("2 ** 8") == "256"
        assert calculator("2 ^ 8") == "256"
        assert calculator("-5 + 15") == "10"
        assert calculator("round(3.14159, 2)") == "3.14"
        assert calculator("sqrt(144)") == "12"
        assert calculator("max(10, 20, 5)") == "20"

    def test_calculator_division_by_zero(self):
        res = calculator("10 / 0")
        assert "Division by zero" in res

    def test_calculator_safety(self):
        # Malicious expressions must be rejected by AST parser
        res1 = calculator("__import__('os').system('ls')")
        assert "Error evaluating expression" in res1

        res2 = calculator("open('/etc/passwd').read()")
        assert "Error evaluating expression" in res2

        res3 = calculator("")
        assert "Error: Empty expression" in res3

    def test_read_local_file(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello from temporary file!")

        content = read_local_file(str(test_file))
        assert content == "Hello from temporary file!"

        # Non-existent file
        missing_content = read_local_file(str(tmp_path / "missing.txt"))
        assert "File not found" in missing_content

    def test_mock_db_lookup(self):
        # User query
        user_res = mock_db_lookup("users")
        users = json.loads(user_res)
        assert isinstance(users, list)
        assert any(u.get("name") == "Alice Johnson" for u in users)

        # Order query
        order_res = mock_db_lookup("orders")
        orders = json.loads(order_res)
        assert isinstance(orders, list)
        assert any(o.get("id") == "ord_501" for o in orders)

        # Product query
        prod_res = mock_db_lookup("products")
        prods = json.loads(prod_res)
        assert isinstance(prods, list)
        assert any("Headphones" in p.get("title", "") for p in prods)

        # Generic query
        generic_res = mock_db_lookup("analytics")
        generic = json.loads(generic_res)
        assert isinstance(generic, list)
        assert generic[0]["query"] == "analytics"

    def test_default_registry(self):
        reg = get_default_registry()
        assert len(reg.list_tools()) == 3
        assert "calculator" in reg.list_tools()
        assert "read_local_file" in reg.list_tools()
        assert "mock_db_lookup" in reg.list_tools()

    def test_tool_payload_windowing_and_telemetry(self):
        reg = ToolRegistry(max_payload_bytes=100)
        
        @reg.register
        def huge_payload_tool():
            """Returns a huge list of records."""
            return [{"id": i, "data": "x" * 50} for i in range(20)]

        res = reg.execute("huge_payload_tool")
        assert res.truncated is True
        assert res.item_count == 20
        assert res.raw_size_bytes is not None and res.raw_size_bytes > 500
        assert "Payload Windowed" in res.output
        assert res.duration_ms >= 0.0
