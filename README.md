# DeepSoul — 下一代 AI Agent 框架

DeepSoul 是一个**会进化的 AI Agent**，它能操控电脑、写代码、管理项目、定时执行任务，并且在每次交互中自动学习变得更聪明。

> 融合 OpenClaw 的编排能力（370K★）+ Hermes Agent 的自我进化（148K★），用 1/3 的代码量实现两者的核心能力。

---

## 能用它做什么

### 日常开发助手
```bash
soul run "帮我创建一个 FastAPI 项目，包含用户登录、JWT 认证、数据库模型"
soul run "重构 soul/engine/agent.py，拆分成多个小方法"
soul run "给这个项目写 Dockerfile 和 docker-compose.yml"
soul run "找出项目中所有的 SQL 注入风险"
```

### 定时自动化
用自然语言描述，自动转为 cron 任务：
```bash
# "每天早上 9 点发日报" → 自动转成 0 9 * * *
soul run "每天早上 9 点检查服务器日志，把错误汇总发到 Telegram"
soul run "每小时备份一次数据库到 /backup"
```

### 跨平台消息网关
启动一次，对接所有聊天平台：
```bash
soul gateway --port 18789
# 然后接入 Telegram / Discord / Slack / 微信...
# 所有平台共享同一个会话状态，Telegram 上聊到一半可以用 CLI 继续
```

### 自我进化
Agent 在完成任务后会自动分析执行过程：
- **学会技能**：成功完成 3 次类似任务 → 自动生成 SKILL.md → 下次直接复用
- **越用越聪明**：GEPA 进化引擎持续优化提示词 → 准确率上升、token 消耗下降
- **记住偏好**：知道你喜欢什么语言、什么框架、什么沟通风格
- **预测需求**：检测到你每天 9 点部署 → 主动提醒并准备好部署检查清单

### 训练自己的 Agent
```bash
# 从一批任务中生成 1000 条训练轨迹 → 用于微调模型
soul train tasks.txt --count 1000 --workers 8
```

---

## 为什么选它而不是 OpenClaw 或 Hermes

| 你要什么 | OpenClaw | Hermes Agent | DeepSoul |
|----------|----------|-------------|----------|
| 开发助手 | ✅ | ✅ | ✅ **+ 预测你下一步要什么** |
| 越用越聪明 | ❌ 纯手写记忆 | ✅ GEPA 进化 | ✅ **GEPA 2.0 + 习惯检测** |
| 同时和多人聊天 | ✅ 20+ 平台 | ⚠️ 6 个平台 | ✅ **WebSocket + REST API** |
| 定时自动干活 | ✅ | ✅ | ✅ **自然语言描述即可** |
| 训练自己的模型 | ❌ | ✅ 复杂 | ✅ **一条命令** |
| 安全可控 | ✅ 沙箱 | ✅ 零遥测 | ✅ **四层护栏 + 审计日志** |
| 安装难度 | 中等 | 低 | **最低 — curl 一键** |
| 一行代码调用 | ❌ 需要 gateway | ❌ | ✅ **API + SDK** |

---

## 5 分钟上手

### 1. 安装
```bash
# 一条命令（自动安装 Python 依赖、创建配置、注册 CLI）
curl -fsSL https://raw.githubusercontent.com/xpailab/deepseek-SOUL/main/scripts/install.sh | bash

# 或者从源码
git clone https://github.com/xpailab/deepseek-SOUL.git
cd deepseek-SOUL
pip install -e ".[all]"
```

### 2. 配置 API Key
```bash
soul config llm.provider deepseek
soul config llm.api_key sk-your-key-here
soul config llm.model deepseek-v4-pro
# 支持 deepseek / claude / openai，写一个就行了
```

### 3. 开始使用
```bash
soul chat                        # 交互式对话（像 ChatGPT 一样用）
soul run "帮我创建 React 项目"    # 单次执行任务
soul status                       # 查看 Agent 状态
soul doctor                       # 系统诊断
```

### 4. 启动网关（可选）
```bash
soul gateway --port 18789
# 浏览器打开 http://localhost:18789 直接使用 Web 聊天界面
# API 文档: http://localhost:18789/docs
```

---

## 它能操控什么

DeepSoul 有内置的工具集，可以直接操作电脑：

| 工具 | 能做什么 | 安全等级 |
|------|---------|---------|
| `bash` | 执行 shell 命令、安装软件、管理进程 | 高风险（有护栏） |
| `file` | 读写编辑删除文件、创建目录 | 中风险（工作空间隔离） |
| `web` | HTTP 请求、网页搜索、API 调用 | 低风险（禁止内网 IP） |

所有工具调用都经过四层安全检查：参数校验 → 路径沙箱 → 命令审批 → 审计记录。

---

## 它是怎么越用越聪明的

```
第一次：soul run "部署 Django 到服务器"
  → 耗时 15 分钟，消耗 8000 tokens
  → 自动分析执行过程...

第二次：soul run "部署 Django 到服务器"
  → 自动匹配到上次学的技能，耗时 5 分钟，消耗 3000 tokens

第十次：soul run "部署 Django 到服务器"
  → GEPA 已优化了 5 代，耗时 2 分钟，消耗 800 tokens
  → Agent 主动问："检测到你每次部署前都会先跑测试，要我把这个加入自动化吗？"
```

三层进化机制：
1. **自动技能生成**：成功完成 3 次类似任务 → 自动创建 SKILL.md 技能文件
2. **GEPA 进化引擎**：帕累托多目标优化（准确率 ↑ + 成本 ↓ + 延迟 ↓），纯 API 无需 GPU
3. **预测记忆**：学习你的习惯 → 提前准备上下文 → 在你开口之前就准备好

---

## 记忆系统 — 四层为什么比三层好

| 层级 | 作用 | 例子 |
|------|------|------|
| L1 冻结快照 | 长期记忆，持久化到文件 | "用户喜欢 Python，讨厌 Java" |
| L2 程序技能 | 自动学会的可复用流程 | "部署 Django 的标准 7 个步骤" |
| L3 FTS5 检索 | 全文搜索历史对话 | "上周那个关于数据库迁移的讨论" |
| L4 预测记忆 | **猜你接下来要做什么** | "你每次改完代码都会跑测试，要我现在跑吗？" |

Hermes 只有前三层。第四层是 DeepSoul 的创新——从被动搜索变成主动预判。

---

## 项目结构

```
deepseek-SOUL/
├── soul/                    # 9,600 行核心代码
│   ├── engine/              # Agent 循环 + Lane Queue 2.0 双层并发调度
│   ├── memory/              # 4 层记忆系统
│   ├── skills/              # 技能自动生成 + GEPA 进化引擎
│   ├── tools/               # 工具系统 + 四层安全护栏
│   ├── prompt/              # Prompt 构建 + 前缀缓存保护 + 上下文压缩
│   ├── llm/                 # DeepSeek / Claude / OpenAI 适配器
│   ├── gateway/             # 统一消息网关 + WebSocket
│   ├── safety/              # Docker 沙箱 + DM 配对 + 审计
│   ├── cron/                # 自然语言定时任务
│   ├── mlops/               # 轨迹生成 → 压缩 → LLM 评估
│  ── config/                # 统一配置管理（YAML + 环境变量）
├── skills/bundled/          # 内置技能（Python 开发、Git 工作流等）
├── web/                     # FastAPI + WebSocket Web UI
├── scripts/install.sh       # 一键安装脚本
├── docker/                  # Docker 容器化
└── pyproject.toml           # Python 项目配置
```

---

## 命令行大全

```bash
soul chat                 # 交互对话
soul chat "帮我写个脚本"   # 单条消息（非交互）
soul run "创建 Python 项目" # 执行单次任务

soul gateway              # 启动网关（端口 18789）
soul config --all         # 查看全部配置
soul config llm.model gpt-4o  # 修改单个配置项
soul status               # 查看运行状态
soul doctor               # 系统诊断（检查依赖、配置、API Key）

soul train tasks.txt --count 1000 --workers 8  # 批量生成训练数据
soul version              # 版本信息
```

---

## API 调用

```python
from soul.engine.agent import Agent
from soul.config.manager import ConfigManager

config = ConfigManager().load()
agent = Agent(config=config)
await agent.initialize()

# 对话
response = await agent.chat("帮我创建一个 Python 项目")
print(response)

# 流式输出
async for chunk in agent.chat_stream("分析这个代码库"):
    print(chunk.content, end="")
```

或者通过 HTTP API：
```bash
curl -X POST http://localhost:18789/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个快速排序"}'
```

---

## 系统要求

- Python 3.11+
- 任意 DeepSeek / Claude / OpenAI API Key
- （可选）Docker（沙箱模式）
- （可选）ripgrep（代码搜索加速）

---

## 许可证

MIT License — 随便用，随便改。
