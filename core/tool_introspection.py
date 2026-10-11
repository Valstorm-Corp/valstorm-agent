"""AST-based Python Function Schema Introspector for Valstorm Dynamic AI Tools."""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple


def _ast_type_to_json_schema(annotation_node: Optional[ast.AST]) -> Dict[str, Any]:
    """Converts a Python AST type annotation node to JSON Schema format."""
    if annotation_node is None:
        return {"type": "string"}

    if isinstance(annotation_node, ast.Name):
        type_id = annotation_node.id.lower()
        if type_id in ("str", "text"):
            return {"type": "string"}
        elif type_id in ("int", "integer"):
            return {"type": "integer"}
        elif type_id in ("float", "number", "decimal"):
            return {"type": "number"}
        elif type_id in ("bool", "boolean"):
            return {"type": "boolean"}
        elif type_id in ("list", "sequence"):
            return {"type": "array"}
        elif type_id in ("dict", "mapping", "object", "any"):
            return {"type": "object"}
        return {"type": "string"}

    elif isinstance(annotation_node, ast.Subscript):
        # Handle List[T], Dict[K, V], Optional[T], Union[T, None]
        base_name = ""
        if isinstance(annotation_node.value, ast.Name):
            base_name = annotation_node.value.id

        slice_node = annotation_node.slice

        if base_name in ("List", "list", "Sequence"):
            inner_schema = _ast_type_to_json_schema(slice_node)
            return {"type": "array", "items": inner_schema}

        elif base_name in ("Dict", "dict", "Mapping"):
            return {"type": "object"}

        elif base_name in ("Optional", "Union"):
            # If Optional[T] or Union[T, None], unwrap inner
            if isinstance(slice_node, ast.Tuple) and slice_node.elts:
                # filter out None
                non_none = [e for e in slice_node.elts if not (isinstance(e, ast.Constant) and e.value is None)]
                if non_none:
                    return _ast_type_to_json_schema(non_none[0])
            return _ast_type_to_json_schema(slice_node)

    return {"type": "string"}


def _parse_docstring_param_descriptions(docstring: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Parses main docstring description and parameter documentation (Google/Sphinx style)."""
    if not docstring or not docstring.strip():
        return "Executes a custom Valstorm dynamic function.", {}

    lines = docstring.strip().splitlines()
    main_desc_lines = []
    param_descs: Dict[str, str] = {}

    in_args_section = False
    current_param = None
    current_param_text = []

    for raw_line in lines:
        line = raw_line.strip()
        if re.match(r"^(Args|Parameters|Arguments|Params):\s*$", line, re.IGNORECASE):
            in_args_section = True
            continue

        if re.match(r"^(Returns|Raises|Yields|Example|Note):\s*$", line, re.IGNORECASE):
            in_args_section = False
            continue

        if in_args_section:
            param_match = re.match(r"^([a-zA-Z0-9_]+)(?:\s*\([^)]+\))?:\s*(.*)$", line)
            if param_match:
                if current_param and current_param_text:
                    param_descs[current_param] = " ".join(current_param_text).strip()
                current_param = param_match.group(1)
                current_param_text = [param_match.group(2).strip()]
            elif current_param and line:
                current_param_text.append(line)
        else:
            if line:
                main_desc_lines.append(line)

    if current_param and current_param_text:
        param_descs[current_param] = " ".join(current_param_text).strip()

    main_desc = " ".join(main_desc_lines).strip() or "Executes a custom Valstorm dynamic function."
    return main_desc, param_descs


def introspect_python_function_code(code_str: str, default_name: str = "custom_function") -> Dict[str, Any]:
    """Inspects Python source code via AST and docstrings to generate an OpenAI/Gemini tool declaration.

    Args:
        code_str: The Python source code containing an 'execute' function or callable definition.
        default_name: The fallback/registered tool identifier.

    Returns:
        A standard tool JSON schema dictionary with 'name', 'description', and 'parameters'.
    """
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        return {
            "name": default_name,
            "description": f"Custom function (syntax error during AST parse: {e})",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    # Find the target function: prefer 'execute', otherwise first top-level function
    target_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "execute":
                target_node = node
                break
            elif target_node is None:
                target_node = node

    if not target_node:
        return {
            "name": default_name,
            "description": "Executes a custom Valstorm dynamic function.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    # Extract docstring & parameter help
    raw_doc = ast.get_docstring(target_node)
    main_desc, param_docs = _parse_docstring_param_descriptions(raw_doc)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    # Map defaults to positional args
    args_list = target_node.args.args
    defaults_list = target_node.args.defaults
    num_defaults = len(defaults_list)
    num_args = len(args_list)
    default_start_idx = num_args - num_defaults

    # Ignore platform injection arguments
    IGNORED_PARAMS = {"platform", "current_user", "self", "cls", "kwargs", "args", "context"}

    for idx, arg_node in enumerate(args_list):
        pname = arg_node.arg
        if pname in IGNORED_PARAMS:
            continue

        prop_schema = _ast_type_to_json_schema(arg_node.annotation)

        # Attach docstring description if present
        if pname in param_docs:
            prop_schema["description"] = param_docs[pname]
        else:
            prop_schema["description"] = f"Argument '{pname}'"

        properties[pname] = prop_schema

        # Check if required (no default value)
        if idx < default_start_idx:
            required.append(pname)

    # Handle kwonlyargs
    kwonlyargs = getattr(target_node.args, "kwonlyargs", [])
    kw_defaults = getattr(target_node.args, "kw_defaults", [])
    for idx, arg_node in enumerate(kwonlyargs):
        pname = arg_node.arg
        if pname in IGNORED_PARAMS:
            continue

        prop_schema = _ast_type_to_json_schema(arg_node.annotation)
        if pname in param_docs:
            prop_schema["description"] = param_docs[pname]
        else:
            prop_schema["description"] = f"Argument '{pname}'"

        properties[pname] = prop_schema

        if idx < len(kw_defaults) and kw_defaults[idx] is None:
            required.append(pname)

    return {
        "name": default_name,
        "description": main_desc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
