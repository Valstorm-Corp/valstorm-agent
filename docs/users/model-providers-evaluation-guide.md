# Next-Gen AI Model Providers Guide for Valstorm Agent (`vsagent`)

If you have primarily been using Google Gemini, this guide covers the top alternative providers, models, local inference engines, and open-weight ecosystems to test and integrate into `vsagent`.

---

## 1. Quick Reference & Cheat Sheet

| Provider | Top Models | OpenAI Base URL | Best Use Case in `vsagent` | Cost vs Gemini |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek** | `deepseek-chat` (V3), `deepseek-reasoner` (R1) | `https://api.deepseek.com/v1` | **Coder & Heavy Reasoner** (Insane benchmark scores, unbeatable price) | ~90% cheaper than Gemini 1.5 Pro |
| **Moonshot / Kimi** | `moonshot-v1-128k`, `kimi-k1.5` | `https://api.moonshot.cn/v1` | **Long-Context Research & Subagents** (2M context, strong long-doc synthesis) | Very economical |
| **Groq** | `llama-3.3-70b-versatile`, `qwen-2.5-coder-32b` | `https://api.groq.com/openai/v1` | **Lightning Subagents & Scouts** (500+ tokens/sec, near-zero latency) | Ultra-cheap / fast |
| **Anthropic** | `claude-3-7-sonnet-20250219`, `claude-3-5-haiku` | *Native SDK adapter* | **Architect & Lead Coder** (Industry gold standard for complex codebases) | Premium tier |
| **Mistral / Codestral** | `codestral-2501`, `mistral-large-2411` | `https://codestral.mistral.ai/v1` | **Monorepo Refactoring & Fill-in-the-Middle** (256k context for code) | Budget-friendly |
| **SiliconFlow / Together** | `deepseek-ai/DeepSeek-V3`, `Qwen/Qwen2.5-Coder-32B` | `https://api.siliconflow.cn/v1` / `https://api.together.xyz/v1` | **Reliable Open-Weight Hosted APIs** with high concurrency | Highly cost-effective |
| **Local (Ollama / vLLM)** | `qwen2.5-coder:32b`, `deepseek-r1:14b` | `http://localhost:11434/v1` | **100% Offline & Zero Telemetry** (No data leaves machine, $0 cost) | **$0.00** |

---

## 2. Deep Dive: Top Providers to Evaluate

### 1. DeepSeek (DeepSeek V3 & R1)
* **Why it matters:** DeepSeek is currently the biggest disruption in the AI space. DeepSeek V3 matches Claude 3.5 Sonnet / GPT-4o across many benchmarks at a fraction of the cost. DeepSeek R1 offers open reasoning comparable to OpenAI o1/o3.
* **Base URL:** `https://api.deepseek.com/v1`
* **Models:**
  * `deepseek-chat`: General-purpose coding, function calling, monorepo edits (DeepSeek-V3).
  * `deepseek-reasoner`: Complex mathematical logic, deep architectural refactoring (DeepSeek-R1).
* **Pricing:** ~$0.14 / 1M input tokens, ~$0.28 / 1M output tokens (cache hits: ~$0.014 / 1M).
* **`vsagent` Setup:**
  ```bash
  export DEEPSEEK_API_KEY="sk-..."
  # In vsagent (with Phase 1 OpenAI spec support):
  vsagent chat --provider deepseek --model deepseek-chat
  ```

---

### 2. Moonshot AI / Kimi (Kimi k1.5 & Moonshot v1)
* **Why it matters:** Kimi is renowned for exceptional long-context fidelity (up to 2 million tokens), multimodal comprehension, and robust tool calling without getting lost in vast document trees.
* **Base URL:** `https://api.moonshot.cn/v1`
* **Models:**
  * `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`
  * `kimi-k1.5` (long-context reasoning)
* **Best `vsagent` Role:** Great for `researcher` subagents scanning massive monorepos or digesting entire API documentation specs.
* **`vsagent` Setup:**
  ```bash
  export MOONSHOT_API_KEY="sk-..."
  ```

---

### 3. Groq (LPU Inference Engine)
* **Why it matters:** Groq runs open-source models on custom LPU chips delivering **300–600 tokens per second**. If you are waiting on agent tool loops, Groq makes the agent feel instantaneous.
* **Base URL:** `https://api.groq.com/openai/v1`
* **Models:**
  * `llama-3.3-70b-versatile`: Generalist logic and tool dispatching.
  * `deepseek-r1-distill-llama-70b`: Fast reasoning on open weights.
  * `qwen-2.5-coder-32b`: Rapid code reviews and syntax checks.
* **Best `vsagent` Role:** Background subagent swarms (`scout`, `backend-tester`, simple regex/file filtering) where raw speed is critical.
* **`vsagent` Setup:**
  ```bash
  export GROQ_API_KEY="gsk_..."
  ```

---

### 4. Mistral AI & Codestral
* **Why it matters:** `codestral-2501` is a dedicated 256k-context model specifically trained on 80+ programming languages. It excels at FIM (Fill-in-the-Middle) and precise code patching.
* **Base URL:** `https://codestral.mistral.ai/v1` (or `https://api.mistral.ai/v1`)
* **Models:**
  * `codestral-latest` (or `codestral-2501`)
  * `mistral-large-latest`
* **Best `vsagent` Role:** Single-file patch generation, unit test creation, and diff reviews.

---

### 5. Local Inference: Ollama & vLLM (Zero-Telemetry Mode)
* **Why it matters:** Complete privacy and zero API costs. Run state-of-the-art open-weight models directly on your Mac (M1/M2/M3/M4 Apple Silicon) or Linux GPU server.
* **Top Recommended Local Coding Models:**
  1. `qwen2.5-coder:32b` (Outstanding coding performance; runs smoothly on 36GB+ Mac unified memory)
  2. `qwen2.5-coder:14b` (Fast, runs smoothly on 16GB–24GB machines)
  3. `deepseek-r1:14b` / `deepseek-r1:32b` (Local reasoning)
* **Ollama Base URL:** `http://localhost:11434/v1`
* **Running Locally:**
  ```bash
  # 1. Install Ollama and pull model
  brew install ollama
  ollama pull qwen2.5-coder:32b

  # 2. Point vsagent to local instance
  vsagent chat --provider ollama --model qwen2.5-coder:32b
  ```

---

### 6. Anthropic Claude (Claude 3.7 Sonnet)
* **Why it matters:** Claude 3.7 Sonnet introduces hybrid reasoning (standard fast execution + dynamic "thinking budget"). It remains the gold standard for full-stack autonomous software engineering in complex codebases.
* **Native SDK Adapter:** Supported directly in `vsagent`.
* **Models:**
  * `claude-3-7-sonnet-20250219`: Primary choice for `architect` and `developer`.
  * `claude-3-5-haiku-20241022`: Super fast, low-cost worker.

---

## 3. Recommended Subagent Swarm Configuration in `vsagent`

By combining these providers, you can build an ultra-efficient, low-cost agent swarm:

```
                               ┌───────────────────────────────────┐
                               │       User Request in CLI         │
                               └─────────────────┬─────────────────┘
                                                 │
                     ┌───────────────────────────▼───────────────────────────┐
                     │          Tier 1: Architect / Lead Coder               │
                     │  Models: Claude 3.7 Sonnet / Gemini 2.5 Pro / R1      │
                     │  Task: High-IQ Planning & Final Code Synthesis        │
                     └─────────────┬───────────────────────────┬─────────────┘
                                   │                           │
                   ┌───────────────▼───────────┐   ┌───────────▼───────────────┐
                   │  Tier 3: Scout / Explorer │   │ Tier 3: Backend Tester    │
                   │  Models: DeepSeek V3 /    │   │ Models: Groq Llama 3.3 /  │
                   │  Ollama Qwen 2.5 Coder    │   │ Ollama Qwen 2.5 Coder     │
                   │  Task: Scan 50+ files,    │   │ Task: Run pytest, parse   │
                   │  grep, AST inspections    │   │ failures, verify builds   │
                   │  Cost: ~$0.00 - $0.14/1M  │   │ Speed: 500+ tokens/sec    │
                   └───────────────────────────┘   └───────────────────────────┘
```

---

## 4. How to Test These Providers in `vsagent`

Once Phase 1 of the `vsagent` v2.0 specification is deployed:

### Set your keys:
```bash
# Cloud Providers
vsagent keys set deepseek <YOUR_DEEPSEEK_KEY>
vsagent keys set groq <YOUR_GROQ_KEY>
vsagent keys set moonshot <YOUR_MOONSHOT_KEY>

# Local Inference
vsagent keys set ollama http://localhost:11434/v1 --type base_url
```

### Run tasks with specific models:
```bash
# 1. Test DeepSeek V3 for monorepo coding
vsagent run --provider deepseek --model deepseek-chat "Inspect apps/api and list all route endpoints"

# 2. Test DeepSeek R1 for reasoning and architecture
vsagent run --provider deepseek --model deepseek-reasoner "Analyze multi-tenant DB isolation in apps/api"

# 3. Test 100% local coding with Ollama (Zero Telemetry)
vsagent chat --provider ollama --model qwen2.5-coder:32b

# 4. Test ultra-fast Groq subagent execution
vsagent run --provider groq --model llama-3.3-70b-versatile "Search repo for all deprecated router usages"
```
