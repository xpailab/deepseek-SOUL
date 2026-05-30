# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

DeepSoul — 下一代 AI Agent 框架，融合 OpenClaw 编排能力 + Hermes 自我进化。
仓库: https://github.com/xpailab/deepseek-SOUL

技术栈: Python 3.11+ / Pydantic v2 / asyncio / aiosqlite / DeepSeek-Claude-OpenAI API / Typer CLI + FastAPI Web UI

## 常用命令

```bash
pip install -e ".[all]"          # 安装所有依赖
soul chat                        # 交互式对话
soul gateway                     # 启动网关 (http://localhost:18789)
soul run "任务描述"               # 单次任务
soul config --all                # 查看配置
soul doctor                      # 诊断检查

pytest tests/ -v                 # 运行全部测试 (17个文件, 205个用例)
pytest tests/test_tools.py -v    # 运行单个测试文件
pytest tests/ -v --cov=soul --cov-report=term-missing  # 带覆盖率

ruff check soul/ tests/          # Lint
mypy soul/                       # 类型检查
```

## 核心架构

### 执行流程（完整链路）

```
用户输入 → LaneQueue(7模式并发调度，默认adaptive)
  → Agent.chat() / chat_stream()
    → 1. 侦察阶段: _recon_prompt() → LLM用只读工具摸底现状 + CWD注入
    → 2. 模糊检测: _is_vague_task() → 任务太模糊则反问1-2个关键问题
    → 3. 记忆检索: MemoryManager.query_for_prompt() → 4层记忆(静态搜索优先，LLM增强仅在无结果时)
    → 4. Prompt组装: PromptBuilder.build_system_prompt() → 11段XML(static/dynamic分离，97%缓存命中)
    → 5. 增强注入: _build_enhanced_prompt() → 角色检测 + 工作记忆 + 计划/纠错 + 检查点续跑
    → 6. LLM推理 → 工具调用循环(max 50轮/100次调用/连续5次失败中止)
    → 7. 每轮: 执行→验证→自纠错→工作记忆更新→检查点保存
    → 8. 任务报告 → 结构化日报 → 教训提取 → 技能学习 → 知识库保存
```

### Agent增强系统（soul/engine/ 的7个增强模块）

全部零额外LLM成本（纯prompt工程+规则+状态追踪）：

| 文件 | 职责 | 关键方法 |
|------|------|---------|
| `personas.py` | 10种内置角色，关键词自动匹配，对话式创建 | `detect_persona()` / `get_persona_prompt()` |
| `working_memory.py` | 会话级工作记忆：已尝试方法、排除方向、错误模式、执行计划、项目文件清单 | `to_prompt()` 生成注入文本, `needs_full_rewrite()` 检测编辑死循环 |
| `verifier.py` | 3层验证：结构化→模式(24种shell错误)→语义 | `verify_tool_result()` / `verify_build_system()` |
| `checkpoint.py` | 长任务断点续跑，1小时内有效 | `save()` / `load_latest()` / `get_resume_context()` |
| `task_stages.py` | 复杂任务拆分为多阶段逐段执行 | `TaskPlan` / `TaskStagePlanner` |
| `lane_queue.py` | 7种队列模式(adaptive/interrupt/steer/collect等)，双层并发(Session+Global) | `LaneQueue` / `SessionLane` |
| `parallel.py` | 多Agent并行：竞速模式(操作) + 收集模式(探索) | `ParallelAgent` |

### 角色系统（soul/engine/personas.py）

10种内置身份，根据任务关键词自动匹配：🧪测试工程师 / 💻开发 / 📊数据分析师 / 📚老师 / 🩺诊断专家 / ✍️文案 / 🧠算法工程师 / 💰金融分析师 / ⚙️DevOps / 🤖通用助手。每个角色自带 skills、context rules、keywords。CLI 支持 `/persona list` 和 `/persona create <name>`，对话中可说"帮我创建一个xxx角色"触发 `persona_creator.skill`。角色上下文在首轮注入 dynamic prompt，不破坏 prefix cache。

### 记忆系统（soul/memory/ 4层 + 错误知识库 + 主动学习）

```
Layer1 FrozenMemory  — 系统prompt快照 + 结构化日报(时间戳/文件清单/状态标记○✓✗)
Layer2 ProceduralMemory — 自动技能生成(≥3次成功→SKILL.md) + 29个内置技能
Layer3 IndexedMemory — FTS5全文搜索 + LLM语义重排(默认关闭，仅静态搜索无结果时启用)
Layer4 PredictiveMemory — 行为预测，主动建议 (SOUL创新)
```

- **错误知识库** (`error_kb.py`): 跨会话累积修复方案，3级匹配(签名哈希→正则→关键词)，quality_score综合评分(置信度×0.5 + 使用频率×0.3 + 时效×0.2)，自动裁剪至200条
- **主动学习管道**: 每次任务→结构化日报+教训提取 → 重复错误≥2次→自动防御规则 → 新会话→注入最近上下文+相关教训
- **教训存储**: `~/.soul/workspace/lessons.jsonl`（100条上限）

### 工具系统（soul/tools/）

5个内置工具(bash/file/web/browser/win)，4层安全防护: `ToolGuardrails`(参数校验→路径沙箱→命令审批→注入检测含20+模式) → `Sandbox`(local/docker/ssh) → `RetryManager`(指数退避+jitter，权限/不存在不重试) → `ResultClassifier`(6种分类)。

文件工具容错: `execute()` 接受 `path`/`file_path` 和 `operation`/`action` 两种参数名，ChatML tokens (`<|im_start|>`) 在文件写入时放行。

### LLM适配（soul/llm/）

`AdapterRegistry` 按provider名称缓存实例。DeepSeekAdapter 支持 reasoning_content(思考模式，流式中保留)，ClaudeAdapter 解析 tool_use 事件，OpenAIAdapter 标准格式。令牌桶限流 + 3次重试(指数退避)。

### Prompt构建（soul/prompt/）

11段XML按序组装: SOUL.md → IDENTITY.md → AGENTS.md → USER.md → MEMORY.md → TOOLS.md → 匹配技能 → 工具声明 → 额外上下文 → 安全规则 → 全局规则(含侦察/自纠错/编码节拍/编译验证/回归检查)。

**Static/Dynamic 分离**: static部分(前9段)放入system prompt享prefix cache，dynamic部分(角色/工作记忆/首轮注入)放在消息末尾。第一轮注入侦察+编码节拍+JSON计划模板，非首轮注入纠错+编译检查+回归检查+错误知识库建议。

`ContextCompressor` 每5轮检查token用量，3种策略: 滑动窗口→中间摘要→LLM智能压缩。

### Gateway（soul/gateway/）

`Gateway` 启动 FastAPI 服务，提供 REST API + WebSocket + Web聊天界面：
- `POST /api/chat` — 同步对话
- `POST /api/sessions` — 创建会话
- `GET /api/status` / `GET /api/sessions` — 状态查询
- `GET /api/audit` / `GET /api/audit/report` — 审计查询
- `POST /webhook/{platform}` — 平台Webhook接收(QQ/微信/钉钉/飞书/Telegram)，含URL验证
- `WS /ws/chat` — WebSocket流式对话，支持steer注入和stop指令

6个平台连接器 (`connectors/`): QQ / WeChat / DingTalk / Feishu / Telegram + `base.py`(PlatformConnector基类)。通过 `register_connector()` 注册并启动监听，`_on_connector_message()` 路由消息到Agent处理。

Web UI 内置单页应用(CHAT_PAGE常量)，含多会话侧边栏(localStorage持久化)、执行状态显示、steer栏、格式化的工具调用展示。

### 其他模块

- **soul/cron/** — 自然语言定时任务，CronScheduler 管理
- **soul/mlops/** — 训练轨迹生成/压缩/LLM评估
- **soul/environments/** — Atropos 环境 + RL 训练
- **soul/safety/** — sandbox(执行隔离) + pairing(DM配对) + auditor(审计记录，已集成到Agent和Gateway)
- **soul/types.py** — 全框架共享的Pydantic模型(Message/ToolCall/ToolResult/SOULConfig等)，修改需谨慎

## 关键设计模式

- **冻结-快照**: Prompt文件在会话开始时snapshot到`~/.soul/workspace/`，保护prefix cache，会话中修改只写盘不更新prompt
- **Static/Dynamic分离**: System prompt永远相同→100%缓存命中，动态内容(角色/工作记忆/纠错)放消息末尾
- **注册表模式**: LLM适配器/Tool/Skill统一通过Registry管理，支持按名称/标签/风险等级查询
- **策略模式**: 7种队列模式、5种重试策略、3种压缩策略，运行时按条件选择
- **Prompt注入防御**: `_sanitize()` 对所有用户文本转义危险标签(20+模式)，但ChatML tokens在文件写入时放行

详细内部流转路径见 `instruction.md`（基于源码逐行确认的 6 阶段完整链路 + "你好" vs "复杂任务"对比）。

## 注意事项

- 不能删除和修改该目录之外的文件
- 使用中文沟通
- 每修改代码后必须运行 `pytest tests/ -v` 验证，17个文件205个用例
- `soul/types.py` 是基础层，修改需谨慎——影响所有模块
- 测试必须兼容 `pytest` (asyncio_mode=auto)，使用 `conftest.py` 的共享fixtures
- 新功能优先通过prompt工程实现，避免增加LLM调用成本
- Agent 的 `chat()` 和 `chat_stream()` 共享 `_chat_prepare()` / `_process_round_results()` 方法，修改循环逻辑时只需改一处
- Web UI 的 JS 内嵌在 Python 字符串中，内层引号用 `&#39;` HTML实体(不用 `\x27`，Python会先处理)
- WebSocket 断开时必须调用 `gen.aclose()` 关闭异步生成器，防止后台空跑烧token
- 文件路径使用 `/` (不是Windows的 `\`)，工作目录不固定
