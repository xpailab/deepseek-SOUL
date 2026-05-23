# DeepSoul — Next-Gen Self-Evolving AI Agent Framework

DeepSoul is an AI agent that **gets smarter with every interaction** — it controls computers, writes code, manages projects, runs scheduled tasks, and learns from every task to improve itself.

> Built with [deepseek-v4-pro](https://platform.deepseek.com) (98%) and [kimi-k2.5](https://platform.moonshot.cn) for multimodal understanding (2%).  
> [中文说明 →](参照.md)

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

### Self-Evolution
After each task, the agent analyzes its execution:
- **Learns skills**: 3+ similar successful tasks → auto-generates SKILL.md → reused next time
- **Gets smarter**: GEPA evolution engine continuously optimizes prompts → accuracy ↑, token cost ↓
- **Remembers preferences**: Learns your preferred languages, frameworks, communication style
- **Predicts needs**: Detects daily patterns (e.g. deploy at 9am) → proactively prepares checklists

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

## Memory System — 4 Layers

| Layer | Function | Example |
|-------|----------|---------|
| L1 Frozen | Long-term memory, persisted to file | "User prefers Python, dislikes Java" |
| L2 Procedural | Auto-learned reusable workflows | "7 standard steps to deploy Django" |
| L3 FTS5 Index | Full-text search over conversation history | "That database migration discussion last week" |
| L4 Predictive | **Anticipates your next move** | "You always run tests after editing code. Run now?" |

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
