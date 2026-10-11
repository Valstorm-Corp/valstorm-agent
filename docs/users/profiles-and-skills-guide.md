# Agent Profiles & Procedural Skills Guide

The Valstorm Agent Runtime features a modular **Profile & Skill Architecture** that equips agents with specialized system prompts, tool whitelists, attached procedural guides, and subagent delegation swarms.

---

## 1. Named Agent Profiles

Profiles define an agent's persona, system instructions, default model, tool permissions, and attached skills.

### Built-in Core Profiles

| Profile Slug | Display Name | Core Specialty | Key Whitelisted Tools |
| :--- | :--- | :--- | :--- |
| **`developer`** | Software Developer | Monorepo coding, refactoring, tests, git workflows. | `terminal_exec`, `patch_file`, `write_file`, `read_file`, `search_files`, `execute_code`, `process_manage`, `skill_view` |
| **`researcher`** | Deep Researcher | Codebase search, architecture exploration, API discovery. | `read_file`, `search_files`, `mock_db_lookup`, `calculator`, `execute_code`, `skill_view`, `skill_list` |
| **`orchestrator`** | Orchestrator Agent | Workflow planning, task breakdown, subagent delegation. | `delegate_task`, `terminal_exec`, `read_file`, `search_files`, `write_file`, `patch_file`, `confirmation_required`, `clarify` |
| **`backend-tester`** | Backend Test Engineer | Automated QA, pytest patterns, failure diagnosis. | `terminal_exec`, `read_file`, `search_files`, `calculator`, `execute_code`, `skill_view` |
| **`writer`** | Technical Writer | Documentation authoring, API specs, architecture guides. | `read_file`, `search_files`, `write_file`, `patch_file`, `skill_view` |
| **`valstorm-assistant`** | Platform Assistant | REST API queries, SQL engine, VFS file search, CRUD. | `valstorm_sql_query`, `valstorm_schema_inspect`, `valstorm_vfs_search`, `valstorm_vfs_browse`, `valstorm_record_cud` |

### Custom User Profiles

Custom profiles are stored as JSON files under `~/.valstorm/profiles/<slug>.json` (or defined in `VALSTORM_PROFILES_DIR`).

Example profile (`~/.valstorm/profiles/security-auditor.json`):
```json
{
  "name": "Security Auditor",
  "api_name": "security-auditor",
  "description": "Inspects code for vulnerabilities, injection flaws, and auth bypasses.",
  "model": "gemini-flash-latest",
  "provider": "gemini",
  "system_prompt": "You are an expert security auditor. Perform thorough static analysis on all target files.",
  "allowed_tools": [
    "read_file",
    "search_files",
    "execute_code",
    "skill_view",
    "skill_list"
  ],
  "attached_skill_slugs": [
    "file-operations-safety",
    "systematic-debugging"
  ],
  "max_turns": 25
}
```

---

## 2. Procedural Skills (170+ Guides)

Skills represent step-by-step Standard Operating Procedures (SOPs) stored under `~/.valstorm/skills/<category>/<slug>/SKILL.md`.

### Skill Tools
Agents can discover and consult skills dynamically during execution:

* **`skill_list(query='...', category='...')`**: Discovers matching skills and lists summaries and tags.
* **`skill_view(name='<slug>', file_path='...')`**: Reads the complete procedure, rules, and reference documentation.

Example interaction:
```python
# The agent discovers relevant skills:
skill_list(query="testing")

# The agent inspects the exact procedure:
skill_view(name="test-driven-development")
```

---

## 3. Subagent Swarm Delegation (`delegate_task`)

The **`orchestrator`** agent (or any authorized profile) can spawn isolated child subagents to execute multi-step workflows without polluting the primary context window.

```python
delegate_task(
    profile="developer",
    goal="Write unit tests for the newly created authentication endpoint in apps/api/auth_test.py",
    context="Endpoint implemented in apps/api/auth.py. Use pytest to verify."
)
```

The subagent executes its own ReAct loop under its profile rules, uses its scoped tools, and returns a structured summary to the orchestrator:

```text
🤖 [SUBAGENT DELEGATION COMPLETE - PROFILE: DEVELOPER]
Child Session ID: aich_sub_92ab81c3d8ef
Profile: Software Developer (developer)
Model: gemini-flash-latest via gemini
Scoped Tools: [terminal_exec, patch_file, write_file, read_file, search_files, calculator, execute_code, process_manage, skill_view, skill_list]
Status: COMPLETED
Duration: 3.42s
Tokens: Prompt=1,420 | Completion=310 | Total=1,730
Tools Executed: [read_file, write_file, terminal_exec]

--- GOAL ---
Write unit tests for the newly created authentication endpoint...

--- SUBAGENT OUTCOME & EXECUTION SUMMARY ---
Successfully authored apps/api/auth_test.py with 4 comprehensive test cases. Executed pytest: 4 passed in 0.08s.
```
