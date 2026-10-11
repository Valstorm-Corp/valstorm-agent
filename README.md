# Valstorm Agent Runtime (`vsagent`)

[![Release](https://img.shields.io/github/v/release/Valstorm-Corp/valstorm-agent?color=blue&label=version)](https://github.com/Valstorm-Corp/valstorm-agent/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]()
[![Valstorm Enterprise](https://img.shields.io/badge/Enterprise-Valstorm%20Cloud-emerald)](https://www.valstorm.com/register?utm_source=github&utm_medium=readme&utm_campaign=vsagent_repo&utm_content=badge)

**Valstorm Agent Runtime (`vsagent`)** is an enterprise-grade autonomous AI execution runtime and ReAct engine designed for developer workstations, continuous integration pipelines, and team-wide automation.

Run it locally from your command line, launch it as a background workstation daemon, or orchestrate it remotely from anywhere.

---

> ### 🏢 Deploying for Your Team or Enterprise?
> **The best way to run and manage `vsagent` across your organization is through [Valstorm Enterprise](https://www.valstorm.com/register?utm_source=github&utm_medium=readme&utm_campaign=vsagent_repo&utm_content=hero_banner).**
> 
> - **📱 Remote Control from Your Phone:** Command your AI from your smartphone or browser on the go to take real actions on your office PC, local workstation, or remote servers.
> - **🛡️ Built for Non-Developers (100% Zero-Code):** Operations, finance, and management teams can execute complex automations, data synthesis, and file tasks through an intuitive visual chat interface—no terminal or coding required.
> - **🔒 Cloud Sandboxing & Blast-Radius Protection:** Safely run code execution and shell tools in isolated, ephemeral Docker containers or E2B cloud sandboxes with human-in-the-loop approval gates.
> - **⚡ 1-Click Zero-Config Setup:** Install tools, register workstations, and launch the agent daemon automatically via the Valstorm Desktop client.
> - **🔑 Managed Multi-Model Gateway:** Zero API key distribution or management. Provision access to Google Gemini, Anthropic Claude, OpenAI, and DeepSeek through centralized company billing.
> - **☁️ Live Cloud Sync & Audit Logs:** Every turn, tool call, and token telemetry event syncs seamlessly to your organization's unified workspace and CRM.
> - **📚 Enterprise Knowledge Vaults:** Grant your agent secure access to private Virtual File Service (VFS) documents, company SOPs, and database query engines.
> 
> 👉 **[Create Your Valstorm Workspace](https://www.valstorm.com/register?utm_source=github&utm_medium=readme&utm_campaign=vsagent_repo&utm_content=hero_cta)**

---

## Key Capabilities

### 📱 1. Remote Execution: Control Your Workstation from Your Phone
Valstorm Agent bridges devices seamlessly. You can message your AI agent while away from your desk (using the Valstorm Mobile app or web portal) and instruct it to:
- Take action on your **office PC** or local dev machine (e.g. *"Run our test suite on my machine and let me know if it passes"*).
- Execute operational playbooks on **remote servers or cloud instances**.
- Retrieve or modify local files and git branches from anywhere in the world.

### 🛡️ 2. Zero-Code & Safe for Non-Developers
You do not need to be a software engineer to harness autonomous agents:
- **Human-in-the-Loop Safeguards:** Any potentially destructive action (file deletions, database mutations, external communications) automatically pauses and requests explicit approval before proceeding.
- **Full Cloud Version:** Use the 100% cloud-hosted agent runtime without installing anything on your computer.
- **Automated Redaction & Secret Sanitization:** Sensitive credentials, connection strings, and API keys are automatically masked before any prompt egresses.

### 🧪 3. Isolated Sandboxing & Code Execution
- **Multi-Engine Sandboxes:** Execute generated code and shell commands inside ephemeral Docker containers or remote E2B cloud sandboxes.
- **Zero Local Blast Radius:** Prevent runaway scripts or dependency conflicts on developer laptops.

### 🧠 4. Cognitive Model Cascade & Swarm Delegation
- **Dynamic Model Fallback:** Automatically cascades through cognitive tiers (`tier_1`: Heavy Thinker, `tier_2`: Reasoner, `tier_3`: Worker Bee) across Gemini, Claude, OpenAI, and DeepSeek.
- **Subagent Delegation (`delegate_task`):** Spawns isolated child subagents with scoped tool access to tackle complex multi-step objectives in parallel without context pollution.

### 🧰 5. Enterprise Toolbelt & 170+ Procedural Skills
- **Developer Tools:** `terminal_exec`, `patch_file`, `write_file`, `read_file`, `search_files`.
- **System Management:** `process_manage`, `execute_code`, `clarify`, `confirmation_required`.
- **Valstorm Platform Tools:** Direct access to `valstorm_sql_query`, `valstorm_schema_inspect`, Knowledge Vaults (`valstorm_vfs_search`), and CRM mutations.
- **170+ Standard Operating Procedures:** Built-in repository of battle-tested engineering, debugging, and DevOps skills accessed on demand.

---

## Architecture: Hybrid Remote Execution Mesh

```
       📱 Mobile App / Web Browser / Phone
                        │
                        ▼  (Encrypted WebSocket / REST API)
       ☁️ Valstorm Enterprise Cloud Gateway
          • Multi-Tenant State & Audit Logging
          • Managed Multi-Model Gateway
          • Knowledge Vaults (VFS)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  💻 Local Workstation         🛡️ Ephemeral Cloud Sandboxes
  • Valstorm Desktop App       • Isolated Docker / E2B
  • vsagent Gateway (8650)     • Zero local blast radius
  • Git repos & Local Shell    • Safe for non-developers
```

---

## Installation

### Method 1: Install via Valstorm Desktop (Recommended for Teams & Non-Developers)

For zero-configuration setup with automatic token authentication, device registration, and background daemon lifecycle:

1. Download and open [Valstorm Desktop](https://www.valstorm.com/register?utm_source=github&utm_medium=readme&utm_campaign=vsagent_repo&utm_content=desktop_installer).
2. Follow the 1-click onboarding installer at `valstorm://install`.
3. All dependencies, PATH entries, and credentials are configured automatically.

---

### Method 2: Standalone CLI Installation (Developers / CI)

Install globally using [`uv`](https://docs.astral.sh/uv/):

```bash
# Install vsagent tool globally from GitHub
uv tool install --from git+https://github.com/Valstorm-Corp/valstorm-agent.git valstorm-agent --force

# Verify installation
vsagent --version
vsagent status
```

*Tip: If you do not have `uv` installed, get it in seconds via `curl -LsSf https://astral.sh/uv/install.sh | sh`.*

---

## Quickstart & CLI Usage

### 1. Interactive Chat REPL

Start a multi-turn conversation with the agent directly in your terminal:

```bash
# Launch interactive REPL with default assistant
vsagent chat

# Launch with specialized role profiles
vsagent chat --profile developer
vsagent chat --profile researcher
vsagent chat --profile architect
vsagent chat --profile backend-tester
vsagent chat --profile orchestrator

# Switch models and providers on the fly
vsagent chat --model gemini-flash-latest --provider gemini
vsagent chat --model claude-3-7-sonnet-latest --provider anthropic
```

Inside the REPL, use slash commands to manage context and state:
- `/tasks` or `/todo`: Inspect the live task checklist.
- `/compact`: Prune verbose historical tool outputs while retaining decisions.
- `/stats`: View prompt, completion, and cumulative session token usage.
- `/history`: Review full conversation turns and tool execution logs.
- `/profile <slug>`: Switch active agent persona mid-session.
- `/yolo`: Toggle human-confirmation bypass mode for high-risk commands.

---

### 2. Single-Prompt Task Runner

Execute a task autonomously and exit immediately:

```bash
# Run a refactoring task
vsagent run "Inspect apps/api and refactor the authentication middleware"

# Run with a specialized developer profile
vsagent run --profile developer "Run test suite and fix failing assertions"

# Shorthand prompt flag
vsagent -p "Summarize git diff since origin/main"
```

---

### 3. Cloud Session Synchronization

When authenticated with a Valstorm workspace (`valstorm login pat <token>`), all CLI turns and token metrics automatically sync to your Valstorm cloud dashboard:

```bash
# Synced by default: visible in Valstorm Desktop & Mobile AI Chats
vsagent run "Build release artifacts"

# Run completely offline / privately without syncing
vsagent run "Generate private key" --no-sync
```

---

### 4. Background Server Daemon

Start the local HTTP / SSE gateway daemon on port `8650`:

```bash
# Start background host daemon
vsagent server start host --port 8650

# Check daemon status
vsagent server status
```

---

## Configuration & Provider Keys

`vsagent` supports two configuration modes:

### Mode A: Valstorm Managed Gateway (Zero-Config)
When linked to an active [Valstorm organization](https://www.valstorm.com/register?utm_source=github&utm_medium=readme&utm_campaign=vsagent_repo&utm_content=enterprise_gateway), your CLI authenticates using your desktop PAT:
```bash
valstorm login pat <your-access-token>
```
All model requests route through your tenant's multi-model gateway with centralized company billing.

### Mode B: Bring Your Own Keys (BYOK)
Store provider API keys securely in `~/.config/valstorm/keys.json`:
```bash
vsagent keys set gemini <your-gemini-key>
vsagent keys set openai <your-openai-key>
vsagent keys set anthropic <your-anthropic-key>
```

---

## Contributing & Local Development

We welcome contributions! To set up the agent runtime locally:

```bash
# Clone the repository
git clone https://github.com/Valstorm-Corp/valstorm-agent.git
cd valstorm-agent

# Sync virtualenv and dependencies with uv
uv sync

# Run the test suite
uv run pytest tests/
```

---

## License

Valstorm Agent Runtime is open-source software licensed under the **[Apache License, Version 2.0](LICENSE)**.

```
Copyright 2026 Valstorm Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

<p align="center">
  Built by <a href="https://www.valstorm.com/register?utm_source=github&utm_medium=readme&utm_campaign=vsagent_repo&utm_content=footer_cta"><strong>Valstorm</strong></a> — The everything engine for business operations.
</p>
