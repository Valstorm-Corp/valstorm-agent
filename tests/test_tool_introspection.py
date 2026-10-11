"""Unit Tests for AST Function Tool Introspection (Phase 2, Task 2.1)."""

import pytest
from core.tool_introspection import introspect_python_function_code


def test_introspect_standard_execute_function():
    code = '''
async def execute(platform, account_id: str, multiplier: float = 1.0, include_tax: bool = False):
    """Calculates monthly recurring revenue and churn metrics.
    
    Args:
        account_id: The target customer account ID.
        multiplier: Optional tier rate multiplier.
        include_tax: Whether to calculate local taxes.
    """
    return {"mrr": 5000}
'''
    schema = introspect_python_function_code(code, default_name="calculate_mrr")
    
    assert schema["name"] == "calculate_mrr"
    assert "Calculates monthly recurring revenue" in schema["description"]
    
    params = schema["parameters"]
    assert params["type"] == "object"
    assert "platform" not in params["properties"]
    
    # account_id
    assert "account_id" in params["properties"]
    assert params["properties"]["account_id"]["type"] == "string"
    assert "customer account ID" in params["properties"]["account_id"]["description"]
    assert "account_id" in params["required"]
    
    # multiplier
    assert "multiplier" in params["properties"]
    assert params["properties"]["multiplier"]["type"] == "number"
    assert "multiplier" not in params["required"]
    
    # include_tax
    assert "include_tax" in params["properties"]
    assert params["properties"]["include_tax"]["type"] == "boolean"
    assert "include_tax" not in params["required"]


def test_introspect_list_and_dict_annotations():
    code = '''
def execute(platform, items: list[str], config: dict):
    """Processes a batch of record items according to configuration."""
    pass
'''
    schema = introspect_python_function_code(code, default_name="batch_process")
    assert schema["name"] == "batch_process"
    assert schema["parameters"]["properties"]["items"]["type"] == "array"
    assert schema["parameters"]["properties"]["config"]["type"] == "object"
    assert "items" in schema["parameters"]["required"]
    assert "config" in schema["parameters"]["required"]


def test_introspect_syntax_error_graceful_handling():
    code = "def execute(bad syntax: "
    schema = introspect_python_function_code(code, default_name="broken_func")
    assert schema["name"] == "broken_func"
    assert "syntax error" in schema["description"].lower()
    assert schema["parameters"]["properties"] == {}
