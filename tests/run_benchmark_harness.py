import os
import sys
import json
import time
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configuration ---
# Ensure you set the BASE_URL to your Agent Runtime instance (e.g., Docker container or local server)
# Reads from environment variable or defaults to local runtime port 8650
BASE_URL = os.environ.get("VALSTORM_API_BASE_URL", "http://localhost:8650/api/v1")
# The PAT is required to authenticate the tool calls against a specific Valstorm organization
ADMIN_PAT = os.environ.get("VALSTORM_ADMIN_PAT") 

if not ADMIN_PAT:
    print("FATAL: VALSTORM_ADMIN_PAT environment variable not set.")
    print("Please set VALSTORM_ADMIN_PAT with an admin token from your isolated benchmark organization.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {ADMIN_PAT}",
    "Content-Type": "application/json"
}

# --- Core API Wrapper ---

def call_tool(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes a direct API call to a Valstorm Agent Tool endpoint.
    """
    url = f"{BASE_URL}/v1/agent/tools/{tool_name}"
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        
        # The API should ideally return token usage and latency metrics in the response body
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"ERROR calling {tool_name}: {e}")
        # Return a structure that signals failure
        return {"error": str(e), "success": False, "result": None}

def api_calculator(expression: str) -> str:
    """Simulates the calculator tool call."""
    response = call_tool("calculator", {"expression": expression})
    # The calculator tool returns the result as a string in its 'result' field
    return str(response.get("result", None))

def api_write_file(path: str, content: str) -> Dict[str, Any]:
    """Simulates the write_file tool call."""
    return call_tool("write_file", {"path": path, "content": content})

def api_read_file(path: str) -> Dict[str, Any]:
    """Simulates the read_file tool call."""
    return call_tool("read_file", {"path": path})

def api_terminal_exec(command: str) -> Dict[str, Any]:
    """Simulates a terminal_exec call for cleanup."""
    return call_tool("terminal_exec", {"command": command})


# --- Benchmark Logic for G-S01 ---

def run_g_s01_benchmark() -> Dict[str, Any]:
    """
    Executes the G-S01 task: calculate a value and write it to a file.
    """
    task_id = "G-S01"
    target_file = "calculation_result.txt"
    expected_content = "18.8495"
    
    metrics = {
        "task_id": task_id,
        "status": "FAIL",
        "message": "Task not completed.",
        "duration_sec": 0.0,
        "start_time": time.time()
    }

    try:
        # Step 1: Perform Calculation (Simulating the agent's first tool call)
        calculation_result = api_calculator(expression="round((42 * 3.14159) / 7, 4)")
        
        if calculation_result != expected_content:
            metrics["message"] = f"Calculation failed. Expected: {expected_content}, Got: {calculation_result}"
            return metrics

        # Step 2: Write the result to the file (Simulating the agent's second tool call)
        api_write_file(path=target_file, content=calculation_result)

        # Step 3: Verification (Read the file back to confirm persistence)
        file_content_response = api_read_file(path=target_file)
        
        # The 'read_file' tool returns content inside a dictionary with metadata
        actual_content = file_content_response.get("content", "").strip()

        if actual_content == expected_content:
            metrics["status"] = "PASS"
            metrics["message"] = "Calculation and file write successful."
        else:
            metrics["message"] = f"Verification failed. File content: '{actual_content}'"

    except Exception as e:
        metrics["message"] = f"Runtime error: {e}"
        metrics["status"] = "ERROR"
    finally:
        metrics["duration_sec"] = time.time() - metrics["start_time"]
        
        # CRITICAL: Clean up the file using the terminal_exec tool
        api_terminal_exec(command=f"rm -f {target_file}")
        
    return metrics

# --- Main Execution ---

if __name__ == "__main__":
    # Ensure a requests-compatible environment like python-dotenv is installed 
    # to automatically load environment variables from the .env file.
    
    print(f"--- Valstorm Agent Benchmark Harness Initial Run ---")
    print(f"Agent Runtime Base URL: {BASE_URL}")

    result = run_g_s01_benchmark()
    
    # Final output of the test result
    print("\n--- G-S01 Test Result ---")
    print(json.dumps(result, indent=4))
    
    if result["status"] == "PASS":
        print("\n✅ Task G-S01 passed successfully.")
    else:
        print(f"\n❌ Task G-S01 failed with status: {result['status']}. See message for details.")