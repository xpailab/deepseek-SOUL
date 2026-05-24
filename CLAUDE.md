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

pytest tests/ -v                 # 运行全部测试
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
    → 1. 侦察阶段: _recon_prompt() → LLM用只读工具摸底现状
    → 2. 模糊检测: _is_vague_task() → 任务太模糊则反问澄清
    → 3. 记忆检索: MemoryManager.query_for_prompt() → 4层记忆
    → 4. Prompt组装: PromptBuilder.build_system_prompt() → 11段XML
    → 5. 增强注入: _build_enhanced_prompt() → 工作记忆+计划+纠错+检查点
    → 6. LLM推理 → 工具调用循环(max 50轮/100次)
    → 7. 每轮: 执行→验证→自纠错→工作记忆更新→检查点保存
    → 8. 任务报告 → 记忆冻结 → 技能学习 → 知识库保存
```

### Agent增强系统（soul/engine/ 的6个增强模块）

这些是按顺序叠加的推理增强层，全部零额外LLM成本（纯prompt工程+规则+状态追踪）：

| 文件 | 职责 | 关键方法 |
|------|------|---------|
| `working_memory.py` | 会话级工作记忆：已尝试方法、排除方向、发现、错误模式、执行计划 | `WorkingMemory.to_prompt()` 生成注入文本 |
| `verifier.py` | 3层验证：结构化(文件存在/非空)→模式(错误关键词)→语义 | `verify_tool_result()` 含18种错误分析 |
| `checkpoint.py` | 长任务断点续跑：每步完成后自动持久化到~/.soul/checkpoints/ | `save()` / `load_latest()` / `get_resume_context()` |
| `task_stages.py` | 复杂任务拆分为多阶段逐段执行，用户确认后继续 | `TaskPlan` / `TaskStagePlanner` |
| `lane_queue.py` | 7种队列模式(含adaptive智能路由)，双层并发(Session+Global) | `LaneQueue` / `SessionLane` |
| `parallel.py` | 多Agent并行：竞速模式(操作) + 收集模式(探索) | `ParallelAgent` |

### 记忆系统（soul/memory/ 4层）

```
Layer1 FrozenMemory  — 系统prompt快照，prefix cache保护
Layer2 ProceduralMemory — 自动技能生成，关键词匹配召回
Layer3 IndexedMemory — FTS5全文搜索 + LLM语义重排
Layer4 PredictiveMemory — 行为预测，主动建议 (SOUL创新)
```

错误知识库 (`error_kb.py`): 跨会话累积修复方案，3级匹配(签名哈希→正则→关键词)，含置信度评分和自动裁剪。

### 工具系统（soul/tools/）

4层安全防护: `ToolGuardrails`(参数校验→路径沙箱→命令审批→Shell隔离) → `Sandbox`(local/docker/ssh) → `RetryManager`(指数退避+jitter) → `ResultClassifier`(6种分类)

### LLM适配（soul/llm/）

`AdapterRegistry` 按provider名称缓存实例，DeepSeekAdapter支持reasoning_content(思考模式)，ClaudeAdapter解析tool_use事件，OpenAIAdapter标准格式。

### Prompt构建（soul/prompt/）

11段XML按序组装: SOUL.md → IDENTITY.md → AGENTS.md → USER.md → MEMORY.md → TOOLS.md → 匹配技能 → 工具声明 → 额外上下文 → 安全规则 → 全局规则(含侦察/自纠错/编译验证)

`ContextCompressor` 在每5轮检查token用量，3种策略: 滑动窗口→中间摘要→LLM智能压缩。

### 类型系统（soul/types.py）

全框架共享的Pydantic模型，包括: Message/ToolCall/ToolResult/StreamChunk/SessionState/SOULConfig及子配置(LLC/Lane/Memory/Skill/Gateway/Sandbox/MLOps)

## 关键设计模式

- **冻结-快照**: Prompt文件在会话开始时snapshot，保护prefix cache，会话中修改只写盘不更新prompt
- **注册表模式**: LLM适配器/Tool/Skill统一通过Registry管理，支持按名称/标签/风险等级查询
- **策略模式**: 7种队列模式、5种重试策略、3种压缩策略，运行时按条件选择
- **Prompt注入防御**: `_sanitize()` 对所有用户文本转义 `<system_reminder>` / `<function_results>` 等危险标签

## 注意事项

- 不能删除和修改该目录之外的文件
- 使用中文沟通
- 每修改代码后必须运行 pytest 验证
- `soul/types.py` 是基础层，修改需谨慎——影响所有模块
- 测试必须兼容 `pytest` (asyncio_mode=auto)，禁止自定义runner
- 新功能优先通过prompt工程实现，避免增加LLM调用成本
- Agent的 `chat()` 和 `chat_stream()` 有大量重复代码，修改循环逻辑时需同步两处
