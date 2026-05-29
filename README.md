# DeepSoul — Next-Gen Self-Evolving AI Agent Framework

> **This project is 99%+ built by [deepseek-v4-pro](https://platform.deepseek.com).**  
> It was created to stress-test whether DeepSeek v4 can autonomously build a production-grade agent framework comparable to OpenClaw. The answer is a clear **yes** — v4's coding capability is exceptional, and combined with its aggressive pricing, the entire project was completed in ~2 weeks (mostly weekends) for **under $7 USD (~¥50 RMB)**.  
> [中文说明 →](README_zh.md)

DeepSoul is an AI agent that **gets smarter with every interaction** — it controls computers, writes code, manages projects, runs scheduled tasks, and learns from every task to improve itself.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What It Can Do

### Daily Development
```bash
deepsoul run "Create a FastAPI project with user login, JWT auth, and DB models"
deepsoul run "Refactor soul/engine/agent.py into smaller methods"
deepsoul run "Write a Dockerfile and docker-compose.yml for this project"
deepsoul run "Find all SQL injection vulnerabilities in this codebase"
```

### Scheduled Automation
Describe tasks in natural language, automatically converted to cron jobs:
```bash
deepsoul run "Every morning at 9am, check server logs and send error summary to Telegram"
deepsoul run "Backup the database to /backup every hour"
```

### Cross-Platform Message Gateway
One gateway connects all chat platforms:
```bash
deepsoul gateway --port 18789
# Supports: QQ / WeChat / DingTalk / Feishu / Telegram / REST API / WebSocket
# All platforms share the same session state
```

### Self-Evolution — Actually Gets Smarter
After each task, the agent **actively learns** and applies lessons to future tasks:
- **Lesson extraction**: Automatically extracts errors, findings, and fixes from every task
- **Defense rules**: Same error pattern ≥2 times → auto-generates proactive warnings for future tasks
- **Structured journaling**: Rich L1 memory entries with timestamps, file lists, findings, and status
- **Active coaching**: New sessions auto-inject relevant past lessons — agent warns you before you repeat mistakes
- **Skill generation**: 3+ similar successful tasks → auto-generates SKILL.md → reused next time
- **GEPA evolution**: Genetic-Pareto optimization continuously improves prompts → accuracy ↑, token cost ↓

### Persona System — 10 Built-in Identities
Auto-detects the right identity for every task and adapts its behavior:
```
You: "Write pytest tests for this module" → 🧪 Test Engineer
You: "Analyze sales trends"              → 📊 Data Analyst
You: "Explain recursion to a beginner"    → 📚 Teacher
You: "Deploy to Kubernetes"               → ⚙️ DevOps
```
Each persona brings its own skills, context rules, and expertise. Say "create a persona for..." to add custom roles.

### Train Your Own Agent
```bash
deepsoul train tasks.txt --count 1000 --workers 8
```

---

## Model Architecture

| Component | Model | Share |
|-----------|-------|-------|
| Core agent loop, reasoning, tool use, code gen | **deepseek-v4-pro** | 98% |
| Multimodal understanding (images, documents) | **kimi-k2.5** | 2% |

DeepSeek v4 Pro provides the backbone — exceptional reasoning at 1/10th the cost of comparable models. Kimi K2.5 handles image and document understanding where needed.

---

## 5-Minute Quick Start

### 1. Install
```bash
curl -fsSL https://raw.githubusercontent.com/xpailab/deepseek-SOUL/main/scripts/install.sh | bash

# Or from source
git clone https://github.com/xpailab/deepseek-SOUL.git
cd deepseek-SOUL
pip install -e ".[all]"
```

### 2. Configure API Key
```bash
deepsoul config llm.provider deepseek
deepsoul config llm.api_key sk-your-key-here
deepsoul config llm.model deepseek-v4-pro
# Supports: deepseek / claude / openai
```

### 3. Start Using
```bash
deepsoul chat                        # Interactive chat
deepsoul run "Create a React project" # Single task execution
deepsoul status                       # Agent status
deepsoul doctor                       # System diagnostics
```

### 4. Launch Gateway (optional)
```bash
deepsoul gateway --port 18789
# Open http://localhost:18789 for Web UI
# API docs: http://localhost:18789/docs
```

---

## Built-in Tools

| Tool | Capability | Risk Level |
|------|-----------|------------|
| `bash` | Shell commands, install software, manage processes | High (guarded) |
| `file` | Read/write/edit/delete files, create directories | Medium (path-sandboxed) |
| `web` | HTTP requests, web search, API calls | Low (no internal IPs) |
| `browser` | Browser automation, web scraping | Medium |
| `win` | Windows GUI automation | Medium |

All tool calls pass 4-layer security: parameter validation → path sandbox → command approval → audit logging.

---

## How It Gets Smarter

```
First time:  deepsoul run "Deploy Django to a server"
  → 15 min, 8000 tokens → auto-analyzes execution...

Second time: deepsoul run "Deploy Django to a server"
  → matches learned skill, 5 min, 3000 tokens

Tenth time:  deepsoul run "Deploy Django to a server"
  → GEPA has optimized 5 generations, 2 min, 800 tokens
  → Agent: "I noticed you always run tests before deploy. Add to automation?"
```

Three-layer evolution:
1. **Auto skill generation**: 3+ successful similar tasks → auto-create SKILL.md
2. **GEPA evolution engine**: Pareto multi-objective optimization (accuracy ↑ + cost ↓ + latency ↓)
3. **Predictive memory**: Learns your habits → prepares context before you ask

---

## Memory System — 4 Layers + Active Learning

| Layer | Function | Enhancement |
|-------|----------|-------------|
| L1 Frozen | Structured journal with timestamps, file lists, findings | Pending tasks (○) preserved, done tasks auto-compressed |
| L2 Procedural | Auto-learned skills + 10 persona-specific skill sets | Persona detection auto-loads relevant skills |
| L3 FTS5 Index | Full-text search with LLM semantic expansion | Smart fallback: static search first (0ms), LLM only when needed |
| L4 Predictive | Behavior prediction from task patterns | — |

**New — Active Learning Pipeline**:
- Each task → structured journal entry + lesson extraction
- Repeated errors → auto defense rules  
- New session → injects recent context + relevant past lessons
- Project file manifest persisted across sessions

Layer 4 is DeepSoul's innovation: from passive recall to proactive prediction.

---

## Architecture

```
deepseek-SOUL/
├── soul/                    # Core engine (~9,600 lines)
│   ├── engine/              # Agent loop + Lane Queue 2.0 concurrent scheduler
│   ├── memory/              # 4-layer memory system
│   ├── skills/              # Auto skill generation + GEPA evolution engine
│   ├── tools/               # Tool system + 4-layer security guardrails
│   ├── prompt/              # Prompt builder + prefix cache + context compression
│   ├── llm/                 # DeepSeek / Claude / OpenAI adapters
│   ├── gateway/             # Unified message gateway + WebSocket
│   ├── safety/              # Docker sandbox + DM pairing + audit
│   ├── cron/                # Natural language scheduled tasks
│   ├── mlops/               # Trajectory generation → compression → LLM evaluation
│   └── config/              # Unified config management (YAML + env vars)
├── skills/bundled/          # 25 built-in skills
├── web/                     # FastAPI + WebSocket Web UI
├── scripts/install.sh       # One-line installer
└── pyproject.toml
```

---

## CLI Reference

```bash
deepsoul chat                  # Interactive chat
deepsoul chat "Write a script" # Single message (non-interactive)
deepsoul run "Create project"  # Execute one-shot task

deepsoul gateway               # Start gateway (port 18789)
deepsoul config --all          # View all config
deepsoul config llm.model gpt-4o  # Change single setting
deepsoul status                # System status
deepsoul doctor                # Diagnostics (deps, config, API key)

deepsoul train tasks.txt --count 1000 --workers 8  # Generate training data
deepsoul version               # Version info
```

---

## API Usage

```python
from soul.engine.agent import Agent
from soul.config.manager import ConfigManager

config = ConfigManager().load()
agent = Agent(config=config)
await agent.initialize()

# Chat
response = await agent.chat("Create a Python project")
print(response)

# Streaming
async for chunk in agent.chat_stream("Analyze this codebase"):
    print(chunk.content, end="")
```

Or via HTTP:
```bash
curl -X POST http://localhost:18789/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a quicksort implementation"}'
```

---

## Requirements

- Python 3.11+
- DeepSeek / Claude / OpenAI API key
- (Optional) Docker for sandbox mode
- (Optional) ripgrep for code search acceleration

---

## License

MIT License — use freely, modify freely.
