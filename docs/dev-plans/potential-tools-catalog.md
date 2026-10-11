# Valstorm Agent Potential Tools Catalog

## Overview & Philosophy

In the Valstorm Agent architecture, **Tools** are strictly code-level, typed Python functions registered into the `ToolRegistry`. Unlike dynamic prompts, tools represent the agent's deterministic interface to the outside world (OS, cloud APIs, browsers, and background processes). 

To ensure absolute enterprise security, prevent prompt injection escapes, and avoid malware, all tools are built into the core agent codebase with explicit input validation, timeout protection, and telemetry tracking.

Based on an analysis of the active tools in your Hermes environment, here is the categorized catalog of potential high-value tools to build into the Valstorm Agent Runtime.

---

## 1. Core Orchestration & Autonomy Tools (Tier 1)

These tools turn a single-turn chatbot into an autonomous multi-agent engineering team.

### A. `delegate_task` (Multi-Agent Swarm Spawner)
* **What it does**: Spawns isolated child subagents running named profiles (`developer`, `researcher`, `backend-tester`) in the background. Each child agent gets its own terminal session, isolated context, and whitelisted toolset.
* **Why it's essential**: Prevents long debugging sessions or massive searches from polluting the orchestrator's context window. The child agent runs independently and returns a clean, consolidated summary.
* **Signature Concept**:
  ```python
  @tool
  async def delegate_task(
      profile: str,
      goal: str,
      context: Optional[str] = None,
      tasks: Optional[List[Dict[str, Any]]] = None,
  ) -> str
  ```

---

### B. `execute_code` (In-Process Python Tool Orchestrator)
* **What it does**: Allows the agent to write and execute a short Python script that calls other tools programmatically in a loop (e.g. `for file in files: read_file(...)`).
* **Why it's essential**: Eliminates the "sluggish agent loop" where the model spends 10 separate roundtrips calling 10 read tools. Instead, it writes 5 lines of Python, processes the batch in a single turn, and reduces token costs by 80%.
* **Signature Concept**:
  ```python
  @tool
  async def execute_code(code: str, timeout_sec: int = 60) -> str
  ```

---

### C. `clarify` (Human-in-the-Loop Decision Dialogs)
* **What it does**: Pauses the agent turn to ask the user a structured question (single-select, multi-select, or open-ended) with recommended options highlighted first.
* **Why it's essential**: Essential when a task has meaningful trade-offs (e.g. *"Should I create a new schema or update existing fields?"*). It stops the agent from making risky assumptions while keeping the UX clean.
* **Signature Concept**:
  ```python
  @tool
  def clarify(
      question: str,
      choices: Optional[List[str]] = None,
      multi_select: bool = False,
  ) -> str
  ```

---

### D. `process` (Persistent Background Process Manager)
* **What it does**: Manages long-running servers or background builds (e.g., `uvicorn`, `yarn dev`, `docker-compose logs`). Actions include `list`, `poll`, `log`, `wait`, and `kill`.
* **Why it's essential**: Terminal commands usually block until exit. `process` allows the agent to spin up a local development server, poll its readiness, and keep working without freezing the conversation.
* **Signature Concept**:
  ```python
  @tool
  async def process_manage(
      action: Literal["list", "poll", "log", "wait", "kill", "submit"],
      session_id: Optional[str] = None,
      data: Optional[str] = None,
  ) -> str
  ```

---

### E. `cronjob` (Autonomous Background Scheduling & Watchdogs)
* **What it does**: Schedules autonomous recurring prompts or scripts (e.g. `every 30m`, `daily at 9am`) that execute in fresh sessions and deliver summaries back to the user or Slack/Teams.
* **Why it's essential**: Powers automated monitoring (e.g., checking for unassigned tasks every hour, watching database health, or running overnight test suites).
* **Signature Concept**:
  ```python
  @tool
  def cronjob_manage(
      action: Literal["create", "list", "update", "pause", "resume", "remove"],
      schedule: Optional[str] = None,
      prompt: Optional[str] = None,
      job_id: Optional[str] = None,
  ) -> str
  ```

---

## 2. Multimodal & Visual Inspection Tools (Tier 2)

These tools give the agent eyes and browser automation capabilities.

### A. `vision_analyze` (Multimodal Image Inspection & Zoom)
* **What it does**: Loads a local image, screenshot, or URL into context. Supports bounding box region cropping (`[x1, y1, x2, y2]`) so the agent can zoom into fine print, UI errors, or diagram details at full resolution.
* **Why it's essential**: Allows developers to say *"Here is a screenshot of the broken UI, fix the CSS"* or analyze wireframes directly.
* **Signature Concept**:
  ```python
  @tool
  async def vision_analyze(
      image_path: str,
      question: str,
      region: Optional[List[int]] = None,
  ) -> str
  ```

---

### B. `browser_exec` (Headless Browser Automation & Scraping)
* **What it does**: Drives a real Playwright / Chrome browser via CDP (Chrome DevTools Protocol) to navigate web pages, click elements, fill inputs, and capture screenshots.
* **Why it's essential**: Allows the agent to verify web apps live, scrape dynamic JavaScript websites, test login flows, or interact with external SaaS portals.
* **Signature Concept**:
  ```python
  @tool
  async def browser_exec(code: str, session: Optional[str] = None) -> str
  ```

---

### C. `text_to_speech` (Voice Synthesis & Audio Notes)
* **What it does**: Converts text responses into high-quality audio files using providers like Deepgram, OpenAI, or ElevenLabs for voice memo playback.
* **Why it's essential**: Allows mobile and desktop voice interactions where the user listens to agent briefings while multitasking.
* **Signature Concept**:
  ```python
  @tool
  async def text_to_speech(text: str, output_path: Optional[str] = None) -> str
  ```

---

## 3. Valstorm Platform Specific Tools (Tier 3)

These tools deeply integrate the agent with Valstorm's enterprise platform engines.

### A. `valstorm_automation_trigger` (Workflow / Celery Trigger)
* **What it does**: Triggers a Valstorm visual automation flow (`/automation`) or background Celery task with custom JSON variables.
* **Why it's essential**: Allows the AI agent to initiate enterprise business workflows (e.g. invoice generation, drip email campaigns, onboarding sequences).
* **Signature Concept**:
  ```python
  @tool
  async def valstorm_automation_trigger(
      automation_id: str,
      payload: Dict[str, Any],
  ) -> str
  ```

---

### B. `valstorm_task_orchestrate` (Server-Side Task Assignment)
* **What it does**: Creates or updates a formal `task` record in the CRM and assigns it to a human team member or another AI agent with review gates.
* **Why it's essential**: Bridges conversational AI with formal team task management and SLA tracking.
* **Signature Concept**:
  ```python
  @tool
  async def valstorm_task_orchestrate(
      task_id: str,
      description: str,
      assigned_to: str,
      status: str = "Request Review",
  ) -> str
  ```

---

### C. `valstorm_git_tracking` (Schema & Blueprint Version Control)
* **What it does**: Commits and pushes Valstorm custom object schemas and metadata changes directly to the organization's linked GitHub/GitLab repository.
* **Why it's essential**: Gives developers automated version control when the agent creates or modifies custom objects and fields.
* **Signature Concept**:
  ```python
  @tool
  async def valstorm_git_tracking(
      commit_message: str,
      objects: Optional[List[str]] = None,
  ) -> str
  ```

---

## 4. Prioritized Implementation Roadmap

| Priority | Tool Name | Target Layer | Primary Value |
| :---: | :--- | :--- | :--- |
| **P1** | **`delegate_task`** | Multi-Agent Swarm | Enables `orchestrator` to spawn parallel `developer` / `researcher` subagents. |
| **P1** | **`execute_code`** | Optimization | Batch tool calls in Python to eliminate sluggish multi-turn agent loops. |
| **P2** | **`clarify`** | Human-in-the-Loop | Safe interactive decision confirmations for high-stakes actions. |
| **P2** | **`process_manage`** | Dev Toolbelt | Run and monitor long-lived dev servers (`yarn dev`, `uvicorn`) in the background. |
| **P3** | **`vision_analyze`** | Multimodal | UI bug screenshot diagnosis and wireframe inspection. |
| **P3** | **`valstorm_automation_trigger`**| Platform Bridge | Connects AI agent decisions directly to Valstorm Celery workflow pipelines. |
