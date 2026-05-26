"""Agent 核心引擎 — 执行循环。

整合:
- LLM 适配器（对话/流式）
- Lane Queue（并发调度）
- Session Manager（会话状态）
- Memory Manager（四层记忆）
- Prompt Builder（提示构建）
- Tool System（工具调用）

主循环:
    用户消息 → Session Lane 入队 → Global Lane 获取槽位
    → 构建 Prompt → LLM 推理 → 工具调用 → 结果处理
    → 记忆更新 → 回复用户 → 释放槽位
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from soul.config.manager import ConfigManager
from soul.engine.checkpoint import CheckpointManager
from soul.engine.lane_queue import LaneQueue, QueueItem
from soul.engine.session import SessionManager
from soul.engine.verifier import ResultVerifier
from soul.engine.working_memory import ExecutionPlan, WorkingMemory
from soul.llm.registry import AdapterRegistry
from soul.memory.error_kb import ErrorKnowledgeBase
from soul.memory.manager import MemoryManager
from soul.prompt.builder import PromptBuilder
from soul.prompt.compressor import ContextCompressor
from soul.tools.builtin.bash import BashTool
from soul.tools.builtin.browser import BrowserTool
from soul.tools.builtin.file import FileTool
from soul.tools.builtin.web import WebTool
from soul.tools.builtin.windows import WindowsTool
from soul.tools.classifier import ResultClassifier
from soul.tools.guardrails import ToolGuardrails
from soul.tools.registry import ToolRegistry
from soul.tools.retry import RetryManager
from soul.types import (
    AgentEvent,
    AgentState,
    MemoryLayer,
    Message,
    MessageRole,
    QueueMode,
    SessionState,
    SOULConfig,
    StreamChunk,
    ToolCall,
    ToolResult,
)


def _build_task_report(
    task: str, rounds: int, results: list, stopped_by_limit: bool, stopped_by_fails: bool
) -> str:
    """生成任务执行报告。"""
    if not results:
        return ""

    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded

    # 简化状态描述
    if stopped_by_limit:
        status = "🛑 已达到执行上限(50轮/100次工具调用)，任务被迫中止"
    elif stopped_by_fails:
        status = "❌ 多次失败导致任务中止"
    elif failed == 0 and succeeded > 0:
        status = "✅ 任务执行成功"
    elif succeeded > 0:
        status = f"⚠️ 任务部分完成 ({succeeded}/{total} 成功)"
    else:
        status = "❌ 任务执行失败"

    # 只显示失败的工具和关键成功的工具
    key_results = []
    for r in results:
        if not r.success:  # 显示失败的
            key_results.append(f"  ✗ {r.name}: {str(r.error or '失败')[:60]}")
        elif r.name in ("file", "write") and r.success:  # 显示文件写入成功
            key_results.append(f"  ✓ {r.name}: 文件写入成功")

    # 限制显示数量
    if len(key_results) > 10:
        key_results = key_results[:10] + [f"  ... 还有 {len(key_results) - 10} 个步骤"]

    lines = [
        "",
        "---",
        "",
        f"**任务总结**: {status}",
        f"执行了 {rounds} 轮对话，调用 {total} 次工具",
    ]

    if key_results:
        lines.append("")
        lines.append("**关键步骤**:")
        lines.extend(key_results)

    return "\n".join(lines)


class Agent:
    """DeepSoul Agent — 核心执行引擎。

    使用示例:
        agent = Agent()
        await agent.initialize()

        # 非流式
        response = await agent.chat("帮我创建一个 Python 项目")

        # 流式
        async for chunk in agent.chat_stream("分析这个代码库"):
            print(chunk.content, end="")
    """

    def __init__(
        self,
        config: SOULConfig | None = None,
        config_path: str | None = None,
    ):
        self.config = config or ConfigManager(config_path).config
        self.cfg_mgr = ConfigManager(config_path)

        # 核心组件（延迟初始化）
        self.llm = AdapterRegistry()
        self.lane_queue = LaneQueue(self.config.lane)
        self.sessions = SessionManager(self.config.memory.workspace_dir)
        self.memory = MemoryManager(self.config.memory)
        self.prompt_builder = PromptBuilder(
            workspace_dir=self.config.memory.workspace_dir,
            skills_dir=self.config.skill.skills_dir,
            soul_file=self.config.soul_file,
            identity_file=self.config.identity_file,
        )
        self.compressor = ContextCompressor()

        # 工具系统
        self.tools = ToolRegistry()
        self.guardrails = ToolGuardrails(self.config.memory.workspace_dir)
        self.classifier = ResultClassifier()
        self.retry_mgr = RetryManager()

        # 安全系统
        from soul.safety.sandbox import Sandbox
        self.sandbox = Sandbox(self.config.sandbox if hasattr(self.config, 'sandbox') else None)

        # 审计系统
        from soul.safety.auditor import Auditor
        self.auditor = Auditor()

        # 保存启动时的 CWD（asyncio/lane queue 可能改变工作目录）
        import os as _os
        self.startup_cwd = _os.getcwd()

        # 工作记忆 + 执行计划（会话级推理增强）
        self.working_memory = WorkingMemory()
        self.verifier = ResultVerifier()

        # 错误知识库（跨会话修复方案积累）
        self.error_kb = ErrorKnowledgeBase()

        # 检查点系统（长任务断点续跑）
        self.checkpoint_mgr = CheckpointManager()

        # 事件系统
        self._event_handlers: dict[str, list[Any]] = {}
        self._initialized = False
        self._running = False

    async def initialize(self) -> None:
        """初始化 Agent — 加载所有子系统。"""
        if self._initialized:
            return

        # 1. 注册内置工具
        self._register_builtin_tools()

        # 2. 初始化记忆系统
        await self.memory.initialize()

        # 3. 初始化会话
        await self.sessions.get_or_create("main")

        # 4. 加载错误知识库
        self.error_kb.load()

        self._initialized = True
        self._running = True

    async def chat(
        self,
        user_message: str,
        session_id: str = "",
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """非流式对话 — 发送消息并获取完整回复。"""
        if not self._initialized:
            await self.initialize()

        session = await self.sessions.get_or_create(session_id=session_id)
        session_id = session.session_id

        # 创建入队消息
        item = QueueItem(
            id=f"msg_{int(time.time() * 1000)}",
            session_id=session_id,
            prompt=user_message,
            mode=QueueMode.ADAPTIVE,
        )

        # 入队并等待处理
        result = await self.lane_queue.enqueue(item)
        acquired = False

        # 如果不是直接 steer，需要从队列取出并获得执行槽位
        if result not in ("steered", "steered_and_queued"):
            item = await self.lane_queue.dequeue(session_id)
            if item is None:
                return "系统繁忙，请稍后重试"
            acquired = True

        try:
            ctx = await self._setup_chat_context(user_message, session_id, system_prompt, tools)
            base_system_prompt = ctx["base_system_prompt"]
            current_messages = ctx["current_messages"]
            saved_len = ctx["saved_len"]
            config = ctx["config"]

            max_rounds = 50
            max_total_tools = 100
            consecutive_fails = 0
            actual_rounds = 0
            all_tool_results: list[ToolResult] = []
            final_content = ""

            for round_num in range(max_rounds):
                # 中循环压缩: 每5轮检查 token 用量，防止工具调用爆窗口
                if round_num > 0 and round_num % 5 == 0:
                    if self.compressor.needs_compression(
                        current_messages, system_tokens=len(system_prompt) // 3
                    ):
                        current_messages = self.compressor.compress(
                            current_messages, system_tokens=len(system_prompt) // 3
                        )
                if len(all_tool_results) >= max_total_tools:
                    final_content += "\n[已达到最大工具调用上限(100次)，任务被迫中止。如需完成请简化任务或分阶段执行]"
                    break
                if consecutive_fails >= 5:  # 增加容错次数，允许复杂任务有更多恢复机会
                    final_content += "\n[连续5次工具执行失败，任务中止。请检查错误原因后重试]"
                    break

                actual_rounds += 1
                await self.sessions.update_state(session_id, AgentState.EXECUTING)

                # 每轮注入最新工作记忆
                live_prompt = self._build_enhanced_prompt(
                    base_system_prompt, user_message, first_round=False
                )

                response = await self.llm.chat(
                    current_messages,
                    tools=tools or self.tools.to_api_schemas(),
                    system_prompt=live_prompt,
                    config=config,
                    provider=config.provider,
                )

                if not response.tool_calls:
                    if response.finish_reason == "length":
                        final_content += response.content
                        current_messages.append(Message(
                            role=MessageRole.ASSISTANT,
                            content=response.content,
                            reasoning_content=response.reasoning_content,
                        ))
                        current_messages.append(Message(
                            role=MessageRole.USER, content="请继续。"
                        ))
                        continue
                    if response.finish_reason == "error":
                        if round_num < max_rounds - 1:
                            await asyncio.sleep(1.5)
                            continue
                    final_content += response.content
                    break

                final_content += response.content  # 工具轮次的文本也要累积

                # 执行工具调用
                round_results: list[ToolResult] = []
                for tc in response.tool_calls:
                    if len(all_tool_results) >= max_total_tools:
                        break
                    tr = await self._execute_tool(tc, session_id)
                    round_results.append(tr)
                    all_tool_results.append(tr)
                    if tr.success:
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1

                # 更新工作记忆（记录尝试、错误、计划进度）
                self._update_working_memory(
                    response.content, round_results, user_message
                )

                # 助手消息（保留 reasoning_content，DeepSeek 思考模式必须传回）
                assistant_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
                current_messages.append(assistant_msg)

                # 工具结果作为独立 TOOL 角色消息追加
                for tr in round_results:
                    tool_msg = Message(
                        role=MessageRole.TOOL,
                        content=str(tr.result)[:4000] if tr.success else (tr.error or "执行失败"),
                        metadata={
                            "tool_call_id": tr.call_id,
                            "tool_name": tr.name,
                        },
                    )
                    current_messages.append(tool_msg)

                # final_content 已经在上面累积了，不要覆盖
                # 继续下一轮循环，让LLM看到工具执行结果

            return await self._finalize_chat(
                user_message, session_id, current_messages, saved_len,
                all_tool_results, actual_rounds, consecutive_fails,
                final_content, max_total_tools,
            )

        except Exception as e:
            error_msg = f"执行任务时出错: {str(e)}"
            self._save_checkpoint(session_id, user_message)
            self.error_kb.save()
            await self.sessions.update_state(session_id, AgentState.IDLE)
            return error_msg

        finally:
            if acquired:
                self.lane_queue.mark_done(session_id)

    async def chat_stream(
        self,
        user_message: str,
        session_id: str = "",
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式对话 — 逐步返回 LLM 输出。"""
        if not self._initialized:
            await self.initialize()

        session = await self.sessions.get_or_create(session_id=session_id)
        session_id = session.session_id

        item = QueueItem(
            id=f"msg_{int(time.time() * 1000)}",
            session_id=session_id,
            prompt=user_message,
            mode=QueueMode.ADAPTIVE,
        )

        result = await self.lane_queue.enqueue(item)
        acquired = False

        if result not in ("steered", "steered_and_queued"):
            item = await self.lane_queue.dequeue(session_id)
            if item is None:
                yield StreamChunk(content="系统繁忙，请稍后重试", finish_reason="error")
                return
            acquired = True

        # 标记流式活跃，使 steer 可以注入
        self.lane_queue.track_active(session_id)

        try:
            ctx = await self._setup_chat_context(
                user_message, session_id, system_prompt, tools,
                state=AgentState.STREAMING,
            )
            base_system_prompt = ctx["base_system_prompt"]
            current_messages = ctx["current_messages"]
            saved_len = ctx["saved_len"]
            config = ctx["config"]
            full_content = ""

            steer_queue: asyncio.Queue[str] = asyncio.Queue()

            async def steer_cb(text: str) -> None:
                await steer_queue.put(text)

            self.lane_queue.register_steer_callback(session_id, steer_cb)

            max_rounds = 50
            max_total_tools = 100
            consecutive_fails = 0
            actual_rounds = 0
            all_tool_results: list[ToolResult] = []

            for round_num in range(max_rounds):
                # 中循环压缩: 每5轮检查 token 用量
                if round_num > 0 and round_num % 5 == 0:
                    if self.compressor.needs_compression(
                        current_messages, system_tokens=len(system_prompt) // 3
                    ):
                        current_messages = self.compressor.compress(
                            current_messages, system_tokens=len(system_prompt) // 3
                        )
                if len(all_tool_results) >= max_total_tools:
                    yield StreamChunk(content="\n[已达到最大工具调用上限(100次)，任务被迫中止。如需完成请简化任务或分阶段执行]")
                    break
                if consecutive_fails >= 5:  # 增加容错次数，允许复杂任务有更多恢复机会
                    yield StreamChunk(content="\n[连续5次工具执行失败，任务中止。请检查错误原因后重试]")
                    break

                actual_rounds += 1
                round_content = ""
                round_reasoning = ""
                round_tool_calls: list[ToolCall] = []
                last_finish = ""

                live_prompt = self._build_enhanced_prompt(
                    base_system_prompt, user_message, first_round=False
                )

                async for chunk in self.llm.chat_stream(
                    current_messages,
                    tools=tools or self.tools.to_api_schemas(),
                    system_prompt=live_prompt,
                    config=config,
                    provider=config.provider,
                ):
                    # 排出 steer 队列
                    while not steer_queue.empty():
                        try:
                            steer_text = steer_queue.get_nowait()
                            yield StreamChunk(content=f"\n[注入: {steer_text}]\n")
                        except asyncio.QueueEmpty:
                            break
                    if chunk.content:
                        round_content += chunk.content
                        full_content += chunk.content
                    if chunk.reasoning_content:
                        round_reasoning += chunk.reasoning_content
                    if chunk.tool_call:
                        round_tool_calls.append(chunk.tool_call)
                    if chunk.finish_reason:
                        last_finish = chunk.finish_reason
                    yield chunk

                # 无工具调用 → 检查是否需要继续
                if not round_tool_calls:
                    if last_finish == "length":
                        current_messages.append(Message(
                            role=MessageRole.ASSISTANT,
                            content=round_content,
                            reasoning_content=round_reasoning,
                        ))
                        current_messages.append(Message(
                            role=MessageRole.USER, content="请继续完成未完成的任务。"
                        ))
                        continue
                    if last_finish == "error" and round_num < max_rounds - 1:
                        continue
                    break

                # 执行工具调用
                round_results: list[ToolResult] = []
                for tc in round_tool_calls:
                    if len(all_tool_results) >= max_total_tools:
                        break
                    tr = await self._execute_tool(tc, session_id)
                    round_results.append(tr)
                    all_tool_results.append(tr)
                    if tr.success:
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1
                    yield StreamChunk(tool_result=tr)

                # 更新工作记忆
                self._update_working_memory(
                    round_content, round_results, user_message
                )

                # 助手消息（保留 reasoning_content）
                assistant_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=round_content,
                    tool_calls=round_tool_calls,
                    reasoning_content=round_reasoning,
                )
                current_messages.append(assistant_msg)

                # 工具结果作为独立 TOOL 消息
                for tr in round_results:
                    tool_msg = Message(
                        role=MessageRole.TOOL,
                        content=str(tr.result)[:4000] if tr.success else (tr.error or "失败"),
                        metadata={
                            "tool_call_id": tr.call_id,
                            "tool_name": tr.name,
                        },
                    )
                    current_messages.append(tool_msg)

            # 生成任务执行报告并 yield（流式特有）
            report = _build_task_report(
                user_message, actual_rounds, all_tool_results,
                len(all_tool_results) >= max_total_tools,
                consecutive_fails >= 5,
            )
            report_chunk = "\n\n" + report
            yield StreamChunk(content=report_chunk)
            full_content = full_content + report_chunk if full_content else report

            if not full_content:
                full_content = "抱歉，任务未能完成。请重试或简化您的请求。"
                yield StreamChunk(content=full_content)

            await self._finalize_chat(
                user_message, session_id, current_messages, saved_len,
                all_tool_results, actual_rounds, consecutive_fails,
                full_content, max_total_tools,
            )

        except Exception as e:
            yield StreamChunk(content=f"\n[执行出错: {str(e)}]", finish_reason="error")
            self._save_checkpoint(session_id, user_message)
            self.error_kb.save()

        finally:
            self.lane_queue.untrack_active(session_id)
            if acquired:
                self.lane_queue.mark_done(session_id)

    def _save_checkpoint(self, session_id: str, task: str = "") -> None:
        """保存当前执行状态为检查点（用于断点续跑）。"""
        try:
            plan = self.working_memory.execution_plan
            plan_steps = []
            for s in plan.steps:
                plan_steps.append({
                    "step": s.step,
                    "action": s.action,
                    "tool": s.tool,
                    "expected": s.expected,
                    "fallback": s.fallback,
                    "completed": s.completed,
                    "success": s.success,
                    "result_summary": s.result_summary,
                })

            self.checkpoint_mgr.save(
                session_id=session_id,
                task=task,
                plan_steps=plan_steps,
                working_memory=self.working_memory,
            )
        except Exception:
            pass  # 检查点保存失败不应影响主流程

    @staticmethod
    def _is_vague_task(text: str) -> bool:
        """检测任务是否过于模糊——需要反问澄清。"""
        t = text.strip()

        # 短确认/回复 → 不反问，由对话历史决定
        confirmations = ["需要", "好的", "好", "行", "可以", "是的", "对", "嗯", "是",
                         "yes", "ok", "yeah", "yep", "sure", "继续", "修复", "开始",
                         "确认", "没问题", "就这样", "搞", "干", "做", "改", "要"]
        if any(t == c for c in confirmations):
            return False

        vague_patterns = ["优化", "改一下", "修一下", "有问题", "不行", "报错", "慢了", "帮我看看"]
        has_specific = any(
            kw in t.lower() for kw in [".py", ".js", ".go", ".ts", "/", "\\", "error",
                "traceback", "log", "日志", "文件", "目录", "端口", "bug", "异常"]
        )
        is_vague = any(p in t for p in vague_patterns)
        if len(t) < 25 and not has_specific and is_vague:
            return True
        return is_vague and not has_specific

    def _recon_prompt(self, task: str, is_multi_turn: bool = False) -> str:
        """生成侦察阶段指令——动手前先摸清现状。"""
        cwd = getattr(self, 'startup_cwd', '') or __import__('os').getcwd()
        ambiguous = self._is_vague_task(task)

        lines = [f"\n## 用户当前工作目录: {cwd}"]

        if is_multi_turn:
            # 多轮对话：不从头侦察，用对话历史做上下文
            lines.append(
                "\n## 多轮对话——不要从头侦察"
                "\n- 这是对话的后续轮次，**对话历史**中已有上文"
                "\n- 先看上一轮你说了什么、做了什么，基于那个上下文理解用户当前消息"
                "\n- 不要再用 file/read 从头侦察目录——除非用户明确要求读新文件"
                "\n- 如果用户在回复你的提问（如「需要」「好的」「对」），直接执行上文讨论的操作"
            )
        else:
            # 首轮：正常侦察流程
            lines.append(
                "\n## 多轮对话注意"
                "\n- 用户可能在延续之前的对话，先回顾**对话历史**，确认是否有相关的上文"
                "\n- 如果用户说「刚才」「之前」「你构建的」等，指的是**对话历史中你刚完成的任务**"
                "\n- 不要从头开始侦察——先看对话历史确认上下文"
            )

        lines.append("\n## 当前阶段: 侦察与理解")
        if is_multi_turn:
            lines.append("这是对话的后续轮次。基于对话历史上文理解用户意图，直接回答或执行。")
        elif ambiguous:
            lines.append("⚠️ 用户的任务描述比较模糊——先回顾对话历史，如果上文已明确则直接回答，不要反问。")
        else:
            lines.append("在制定计划之前，先用 1-2 个只读工具快速摸底。")
        lines.append("侦察后简要总结发现，然后制定执行计划并开始执行。")

        return "\n".join(lines)

    @staticmethod
    def _build_verify_prompt(tool_name: str, filepath: str = "") -> str:
        """为代码修改生成编译/运行验证提示。"""
        code_extensions = {
            ".py": "python -m py_compile <file> 或 python <file>",
            ".js": "node --check <file>",
            ".ts": "npx tsc --noEmit <file>",
            ".go": "go build ./...",
            ".rs": "cargo check",
            ".java": "javac <file> 或 mvn compile",
            ".c": "gcc -Wall -o /dev/null <file>",
            ".cpp": "g++ -Wall -o /dev/null <file>",
            ".sh": "bash -n <file>",
        }

        if not filepath:
            return ""

        ext = filepath[filepath.rfind("."):].lower() if "." in filepath else ""
        check_cmd = code_extensions.get(ext, "")
        if not check_cmd:
            return ""

        check_cmd = check_cmd.replace("<file>", filepath)
        return (
            f"\n[编译验证] 刚修改了代码文件 {filepath}，请运行编译/语法检查:\n"
            f"  {check_cmd}\n"
            f"  如果检查失败，立即修复错误后再继续。"
        )

    # ═══════════════════════════════════════════
    # 编码自动化行为：强制验证 + 小步快跑 + 回归检查
    # ═══════════════════════════════════════════

    @staticmethod
    def _is_coding_task(text: str) -> bool:
        """检测是否为编码/开发类任务。"""
        coding_keywords = [
            "写", "创建", "开发", "实现", "修改", "重构", "改", "加",
            "代码", "函数", "类", "模块", "接口", "API", "api",
            ".py", ".js", ".go", ".rs", ".java", ".ts",
            "app", "service", "controller", "model", "route",
        ]
        return any(kw in text for kw in coding_keywords)

    def _coding_cadence_prompt(self, task: str) -> str:
        """生成"小步快跑"编码节拍指令。"""
        if not self._is_coding_task(task):
            return ""
        return (
            "\n## 编码节拍规则 — 小步快跑\n"
            "你是开发者，不是打字员。必须遵守以下节奏：\n"
            "1. 写一小段代码（一个函数/一个类/一个文件）→ 立即用工具验证\n"
            "2. 验证通过 → 写下一段\n"
            "3. 验证失败 → 马上看错误 → 修正 → 再验证 → 通过才继续\n"
            "4. 不要连续写 3 个文件才测一次——每个文件写完就测\n"
            "5. 不要猜测代码能不能跑——跑一下就知道\n"
            "6. 完成后运行项目已有的测试套件（pytest/go test/npm test）确认没破坏已有功能"
        )

    def _coding_guard(self, round_results: list) -> str:
        """强制编译检查——检测到代码写入后注入验证指令。"""
        code_writes = []
        for tr in round_results:
            if not hasattr(tr, 'success') or not tr.success:
                continue
            if tr.name not in ("write_file", "write", "edit_file", "edit", "file"):
                continue
            # 获取写入的文件路径
            filepath = ""
            if hasattr(tr, 'result'):
                if isinstance(tr.result, dict):
                    filepath = tr.result.get("path", tr.result.get("file_path", ""))
                elif isinstance(tr.result, str):
                    filepath = tr.result

            ext = filepath[filepath.rfind("."):].lower() if "." in filepath else ""
            if ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".sh"):
                code_writes.append(filepath)

        if not code_writes:
            return ""

        # 生成强制验证指令
        cmds = []
        for fp in code_writes[-3:]:  # 最多3个文件
            cmd = self._build_verify_prompt("write_file", fp)
            if cmd:
                cmds.append(cmd)

        if not cmds:
            return ""

        return (
            "\n## ⚠️ 刚修改了代码文件 — 必须立即验证\n"
            + "\n".join(cmds)
            + "\n\n不要跳过这一步。不要继续写下一个文件。先运行上面的检查命令。"
        )

    def _regression_guard(self) -> str:
        """回归检查——任务快完成时强制全量测试。"""
        plan = self.working_memory.execution_plan
        if plan.is_empty():
            return ""

        total = len(plan.steps)
        completed = sum(1 for s in plan.steps if s.completed)
        # 80%完成时触发
        if total < 3 or completed < max(2, int(total * 0.8)):
            return ""

        # 只在首次触发时注入
        if self.working_memory.has_tried("回归测试"):
            return ""

        self.working_memory.record_attempt("回归测试", success=False,
            result="待执行", tool="_regression")

        return (
            "\n## 回归检查 — 任务接近完成\n"
            "大部分步骤已完成。在报告'任务完成'之前，必须运行一次完整验证：\n"
            "- 如果有 Makefile: 运行 make test 或 make check\n"
            "- 如果有 pyproject.toml: 运行 pytest 或 python -m pytest\n"
            "- 如果有 package.json: 运行 npm test\n"
            "- 如果有 go.mod: 运行 go test ./...\n"
            "- 如果有 Cargo.toml: 运行 cargo test\n"
            "- 至少运行新增/修改文件的编译检查和语法检查\n"
            "确认全部通过后才可以说'任务完成'。如果测试失败，修复后再宣告完成。"
        )

    def _coding_guard_from_memory(self) -> str:
        """从工作记忆中检查本轮是否写了代码 → 强制验证。"""
        code_writes = self.working_memory.code_writes
        if not code_writes:
            return ""

        cmds = []
        for fp in code_writes[-3:]:
            cmd = self._build_verify_prompt("write_file", fp)
            if cmd:
                cmds.append(cmd)
        if not cmds:
            return ""

        self.working_memory.code_writes.clear()

        return (
            "\n## ⚠️ 刚修改了代码文件 — 必须立即验证再继续\n"
            + "\n".join(cmds)
            + "\n\n❗ 先跑上面的检查。通过了再写下一个文件。不要跳过。"
        )

    # ═══════════════════════════════════════════
    # 问题解决增强：计划 + 工作记忆 + 自纠错 + 验证
    # ═══════════════════════════════════════════

    def _build_enhanced_prompt(
        self, base_prompt: str, user_message: str, first_round: bool,
        is_multi_turn: bool = False,
    ) -> str:
        """构建增强系统提示——注入工作记忆、错误知识库、检查点。"""
        wm = self.working_memory
        parts = [base_prompt]

        # 注入工作记忆上下文
        wm_text = wm.to_prompt()
        if wm_text:
            parts.append(wm_text)

        # 首轮：注入首轮专用规则 + 侦察指令 + 检查点续跑或规划
        if first_round:
            if not is_multi_turn:
                # 首轮专用规则（侦察/反问/编辑策略）——从 builder 注入
                parts.append(self.prompt_builder._first_round_injection())
            if wm.execution_plan.is_empty():
                cp = self.checkpoint_mgr.load_latest(max_age_hours=1)
                if cp:
                    resume_context = self.checkpoint_mgr.get_resume_context(cp)
                    parts.append(resume_context)
                    wm.execution_plan = ExecutionPlan(task=cp.task)
                    for s in cp.plan_steps:
                        from soul.engine.working_memory import PlanStep
                        step = PlanStep(
                            step=s.get("step", 0),
                            action=s.get("action", ""),
                            completed=s.get("completed", False),
                            success=s.get("success"),
                            result_summary=s.get("result_summary", ""),
                        )
                        wm.execution_plan.steps.append(step)
                    for f in cp.findings:
                        wm.add_finding(f)
                    for r in cp.ruled_out:
                        wm.rule_out(r)
                elif not is_multi_turn:
                    # 侦察阶段指令：动手前先看清楚（仅首轮）
                    parts.append(self._recon_prompt(user_message, is_multi_turn=False))
                    # 编码任务：注入小步快跑节拍
                    coding_cadence = self._coding_cadence_prompt(user_message)
                    if coding_cadence:
                        parts.append(coding_cadence)
                    # 规划指令
                    parts.append(wm.get_planning_prompt(user_message))

        # 逐行修补死循环检测：同一文件 3+ 次编辑仍失败 → 强制全文重写
        if not first_round:
            rewrite_file = wm.needs_full_rewrite()
            if rewrite_file:
                parts.append(wm.get_rewrite_prompt(rewrite_file))

        # 强制编码验证：上轮写了代码就注入检查指令
        if not first_round:
            code_guard = self._coding_guard_from_memory()
            if code_guard:
                parts.append(code_guard)

        # 回归检查（任务快完成时注入）
        regression = self._regression_guard()
        if regression:
            parts.append(regression)

        # 错误知识库建议（当最近有失败时）
        last_err = wm.last_error()
        if last_err and not last_err.get("fix"):
            kb_entry = self.error_kb.lookup_by_confidence(
                last_err["error"], min_confidence=0.5
            )
            if kb_entry:
                parts.append(
                    f"\n[知识库建议] 历史记录中相似的错误修复方案:\n"
                    f"  错误特征: {kb_entry.root_cause[:150]}\n"
                    f"  已知修复: {kb_entry.fix[:200]}\n"
                    f"  历史成功率: {kb_entry.confidence:.0%}"
                )

        # 连续失败：注入纠错指令
        correction = wm.get_correction_prompt()
        if correction:
            parts.append(correction)

        return "\n\n".join(parts)

    def _update_working_memory(
        self,
        response_content: str,
        round_results: list,
        user_message: str,
    ):
        """从本轮 LLM 响应和工具结果更新工作记忆。"""
        wm = self.working_memory

        # 尝试从响应中提取执行计划
        if wm.execution_plan.is_empty():
            plan = ExecutionPlan.parse_from_text(response_content, user_message)
            if not plan.is_empty():
                wm.set_plan(plan)

        # 记录每步工具执行结果 + 验证 + 错误知识库
        for tr in round_results:
            if hasattr(tr, 'success'):
                wm.record_attempt(
                    action=f"{tr.name}",
                    tool=tr.name,
                    result=str(tr.result)[:200] if tr.result else "",
                    success=tr.success,
                )
                if not tr.success and tr.error:
                    # 查询错误知识库——是否有已知修复方案
                    kb_entry = self.error_kb.lookup(tr.error, tool=tr.name)
                    diagnosis = kb_entry.root_cause if kb_entry else ""
                    fix = kb_entry.fix if kb_entry else ""

                    wm.record_error(
                        tool=tr.name,
                        error=tr.error,
                        diagnosis=diagnosis,
                        fix=fix,
                    )
                elif tr.success:
                    # 上一步失败、这一步成功 → 知识库学习
                    last_err = wm.last_error()
                    if last_err and last_err["tool"] == tr.name:
                        self.error_kb.learn(
                            error_text=last_err["error"],
                            tool=tr.name,
                            fix=f"成功方案: {str(tr.result)[:200]}",
                        )
                        self.error_kb.record_result(last_err["error"], True)

                # 结果验证：即使 success=True，也检查输出质量
                expected = ""
                plan = wm.execution_plan
                if not plan.is_empty() and plan.current_step():
                    expected = plan.current_step().expected

                vr = self.verifier.verify_tool_result(
                    tool_name=tr.name,
                    result=tr.result if hasattr(tr, 'result') else None,
                    error=tr.error if hasattr(tr, 'error') else "",
                    expected=expected,
                )
                wm.record_verification(
                    tool=tr.name,
                    passed=vr.passed,
                    issues=vr.issues,
                    suggestions=vr.suggestions,
                )

                # 验证失败但工具报告成功 → 修正 success 标记
                if not vr.passed and tr.success and vr.severity == "error":
                    wm.add_finding(f"{tr.name} 虽然返回成功但验证发现问题: {'; '.join(vr.issues[:2])}")

                # 编译/运行验证：修改代码文件后追踪并强制检查
                if tr.success and tr.name in ("write_file", "write", "edit_file", "edit", "file"):
                    filepath = ""
                    if hasattr(tr, 'result') and isinstance(tr.result, dict):
                        filepath = tr.result.get("path", tr.result.get("file_path", ""))
                    if filepath:
                        ext = filepath[filepath.rfind("."):].lower() if "." in filepath else ""
                        if ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".sh"):
                            wm.code_writes.append(filepath)
                            verify_prompt = self._build_verify_prompt(tr.name, filepath)
                            if verify_prompt:
                                wm.add_finding(verify_prompt.strip())

        # 标记计划步骤完成
        plan = wm.execution_plan
        if not plan.is_empty():
            current = plan.current_step()
            if current and round_results:
                last_tr = round_results[-1] if round_results else None
                if last_tr and hasattr(last_tr, 'success'):
                    current.mark_done(
                        last_tr.success,
                        str(last_tr.result)[:200] if last_tr.result else last_tr.error or "",
                    )

        # 检测连续失败 → 自动标记当前方向为"已排除"
        if wm.repeated_failures(2):
            if round_results:
                failed_names = [
                    tr.name for tr in round_results
                    if hasattr(tr, 'success') and not tr.success
                ]
                if failed_names:
                    wm.rule_out(f"{', '.join(failed_names)} 方向")

    async def _setup_chat_context(
        self, user_message: str, session_id: str,
        system_prompt: str, tools: list[dict] | None,
        state: AgentState = AgentState.THINKING,
    ) -> dict:
        """chat/chat_stream 公共前置处理——记忆检索+Prompt构建+循环变量初始化。

        多轮对话时跳过重开销操作（记忆检索/技能匹配/侦察注入），
        直接用对话历史作为上下文。"""
        await self.sessions.update_state(session_id, state or AgentState.THINKING)
        history = await self.sessions.get_history(session_id)
        is_multi_turn = len(history) >= 2  # 至少一轮完整对话

        if is_multi_turn:
            # 多轮对话：极简 prompt——对话历史已包含全部上下文
            import platform
            base_system_prompt = (
                "<agent_rules>\n"
                "- 你是 DeepSoul，使用中文与用户沟通\n"
                "- 对话历史中已有之前的完整上下文，直接基于历史回答\n"
                "- 当前系统: " + platform.system() + "\n"
                "- 遇到错误时分析根因，不要盲目重试\n"
                "- 直接做事，完成后简要总结\n"
                "</agent_rules>"
            )
            if system_prompt:
                base_system_prompt = base_system_prompt + "\n\n" + system_prompt
            enhanced_prompt = base_system_prompt  # 不注入额外内容
            self.working_memory.clear()
        else:
            # 首轮：完整管线
            memory_context = await self.memory.query_for_prompt(user_message)
            matched_skills = self.memory.procedural.match(user_message, top_k=2)
            base_system_prompt = self.prompt_builder.build_system_prompt(
                matched_skills=matched_skills,
                tools=tools or self.tools.to_api_schemas(),
                extra_context=memory_context,
            )
            if system_prompt:
                base_system_prompt = base_system_prompt + "\n\n" + system_prompt
            enhanced_prompt = self._build_enhanced_prompt(
                base_system_prompt, user_message, first_round=True,
                is_multi_turn=False,
            )
            self.working_memory.clear()

        user_msg = Message(role=MessageRole.USER, content=user_message)
        full_messages = self.prompt_builder.build_messages(
            history + [user_msg],
            system_prompt=enhanced_prompt,
            tools=tools or self.tools.to_api_schemas(),
        )
        return {
            "base_system_prompt": base_system_prompt,
            "current_messages": list(full_messages),
            "saved_len": len(full_messages),
            "config": self.config.llm,
        }

    async def _finalize_chat(
        self, user_message: str, session_id: str,
        current_messages: list, saved_len: int,
        all_tool_results: list, actual_rounds: int,
        consecutive_fails: int, final_content: str,
        max_total_tools: int,
    ) -> str:
        """chat/chat_stream 公共后处理——报告+持久化+技能学习+检查点。"""
        report = _build_task_report(
            user_message, actual_rounds, all_tool_results,
            len(all_tool_results) >= max_total_tools,
            consecutive_fails >= 5,
        )
        final_content = final_content + "\n\n" + report if final_content else report

        for msg in current_messages[saved_len:]:
            await self.sessions.add_message(session_id, msg)

        if not final_content:
            final_content = "抱歉，任务未能完成。请重试或简化您的请求。"

        await self.memory.observe_action(user_message)
        await self.memory.store_conversation(session_id, current_messages)
        summary = f"用户: {user_message[:100]} | 回复: {final_content[:100]}"
        await self.memory.remember(summary, layer=MemoryLayer.FROZEN)

        is_simple = (actual_rounds == 1 and len(all_tool_results) == 0)
        if not is_simple:
            if self.config.skill.auto_generate and consecutive_fails < 5:
                await self._learn_from_task(user_message, current_messages, all_tool_results)
            if consecutive_fails < 5:
                self.checkpoint_mgr.mark_complete(session_id)
            else:
                self._save_checkpoint(session_id, user_message)
            self.error_kb.save()

        await self.sessions.update_state(session_id, AgentState.IDLE)
        return final_content

    async def _execute_tool(
        self, tool_call: ToolCall, session_id: str = ""
    ) -> ToolResult:
        """执行工具调用（带安全检查 + 审计记录）。"""
        tool_def = self.tools.get(tool_call.name)
        if not tool_def:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=f"未知工具: {tool_call.name}",
                classification="failure",
            )

        # 安全检查
        is_safe, reason = self.guardrails.check_tool_call(
            tool_call.name, tool_call.arguments, tool_def.risk
        )
        if not is_safe:
            self.auditor.record_safety_block(
                tool_name=tool_call.name,
                reason=reason,
                arguments=tool_call.arguments,
            )
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=f"安全检查失败: {reason}",
                classification="denied",
            )

        # 高风险操作需确认（交互模式中应提示用户）
        if tool_def.requires_approval:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error="此操作需要用户确认（当前不支持自动审批）",
                classification="denied",
            )

        # 执行（带异常保护 + 沙箱隔离 bash/exec 类工具）
        start = time.time()
        try:
            # bash/shell/exec 工具用沙箱隔离执行
            if tool_call.name in ("bash", "shell", "exec", "run"):
                command = tool_call.arguments.get("command", "")
                timeout = tool_def.timeout_seconds
                sandbox_result = await self.sandbox.execute(
                    command=command,
                    mode="local",
                    timeout=timeout,
                )
                result = sandbox_result.get("stdout", "")
                error = sandbox_result.get("stderr", "")
                retries = 0
                self.auditor.record_file_access(
                    filepath=f"sandbox:{tool_call.name}",
                    action="sandbox_execute",
                    session_id=session_id,
                )
            else:
                result, error, retries = await self.retry_mgr.execute_with_retry(
                    tool_def.handler,
                    **tool_call.arguments,
                    tool_name=tool_call.name,
                )
        except Exception as e:
            result = None
            error = str(e)

        elapsed = (time.time() - start) * 1000

        tool_def.call_count += 1
        if error:
            tool_def.error_count += 1

        # 审计记录
        self.auditor.record_tool_call(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            result=str(result)[:500] if result else error or "",
            session_id=session_id,
        )

        return self.classifier.classify(
            tool_call.name,
            result,
            error=error,
            duration_ms=elapsed,
            timeout_seconds=tool_def.timeout_seconds,
            call_id=tool_call.id,
        )

    def _register_builtin_tools(self) -> None:
        """注册内置工具。"""
        import platform
        self.tools.register(BashTool.to_tool_def())
        self.tools.register(FileTool.to_tool_def())
        self.tools.register(WebTool.to_tool_def())
        self.tools.register(BrowserTool.to_tool_def())
        if platform.system() == "Windows":
            self.tools.register(WindowsTool.to_tool_def())

    # ═══════════════════════════════════════════
    # 事件系统
    # ═══════════════════════════════════════════

    def on(self, event_type: str, handler: Any) -> None:
        """注册事件处理器。"""
        self._event_handlers.setdefault(event_type, []).append(handler)

    async def _emit(self, event: AgentEvent) -> None:
        """触发事件。"""
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

    async def _learn_from_task(
        self,
        task: str,
        messages: list[Message],
        tool_results: list[ToolResult],
    ) -> None:
        """从成功任务中自动学习 — 连接 SkillGenerator + GEPA 进化管道。

        当 auto_generate=True 时，成功任务结束后自动:
        1. 构建执行追踪 (execution trace)
        2. 调用 ProceduralMemory.create_from_trace() 生成/更新技能
        3. 若 gepa_enabled=True，加载 GEPAEngine 进化技能
        """
        try:
            # 构建执行追踪
            trace: list[dict[str, Any]] = []
            for msg in messages:
                if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                    for tc in msg.tool_calls:
                        trace.append({
                            "action": "tool_call",
                            "description": f"{tc.name}: {str(tc.arguments)[:200]}",
                        })
                elif msg.role == MessageRole.TOOL:
                    trace.append({
                        "action": "tool_result",
                        "description": (msg.content or "")[:200],
                    })
                elif msg.role == MessageRole.ASSISTANT and msg.content:
                    trace.append({
                        "action": "decision",
                        "description": msg.content[:200],
                    })

            if len(trace) < 2:
                return  # 太短的任务不值得学习

            # Layer 2: 程序性记忆 — 从追踪生成/更新技能
            skill = await self.memory.learn_skill(
                task=task,
                trace=trace,
                success=all(r.success for r in tool_results) if tool_results else True,
                tool_results=tool_results,
            )
            if skill is None:
                return

            # GEPA 进化 — 优化已生成的技能
            if self.config.skill.gepa_enabled:
                try:
                    from soul.skills.gepa import GEPAEngine
                    engine = GEPAEngine(
                        skills_dir=self.config.skill.skills_dir,
                        max_generations=self.config.skill.gepa_generations,
                        population_size=self.config.skill.gepa_population,
                    )
                    evolved = await engine.evolve(
                        skill,
                        evaluator=None,  # 使用默认评估器
                        task_context=task,
                    )
                    if evolved:
                        skill = evolved
                except ImportError:
                    pass  # GEPA 依赖不可用时静默跳过

        except Exception:
            pass  # 技能学习失败不影响主流程

    # ═══════════════════════════════════════════
    # 管理 API
    # ═══════════════════════════════════════════

    async def compact(self, session_id: str = "") -> None:
        """压缩会话上下文。"""
        if session_id:
            session = await self.sessions.get(session_id)
            if session:
                compressed = self.compressor.compress(session.messages)
                session.messages = compressed
        else:
            for state in (await self.get_all_sessions()):
                state.messages = self.compressor.compress(state.messages)

    async def get_status(self) -> dict[str, Any]:
        """获取 Agent 状态。"""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "llm": {
                "provider": self.config.llm.provider,
                "model": self.config.llm.model,
                "usage": self.llm.get(self.config.llm).get_usage_stats(),
            },
            "sessions": {
                "active": self.sessions.active_count,
                "list": await self.sessions.list_sessions(),
            },
            "lane_queue": self.lane_queue.get_stats(),
            "memory": await self.memory.get_stats(),
            "tools": self.tools.get_stats(),
        }

    async def shutdown(self) -> None:
        """关闭 Agent。"""
        self._running = False
        self.auditor.flush()
        self.error_kb.save()
        if self._initialized:
            await self.sessions.close_all()
            await self.memory.close()
            await self.llm.close_all()

    async def get_all_sessions(self) -> list[SessionState]:
        """获取所有会话状态。"""
        return [
            state for state in self.sessions._sessions.values()
        ]

    # ═══════════════════════════════════════════
    # 聊天命令
    # ═══════════════════════════════════════════

    async def handle_command(self, command: str, session_id: str) -> str:
        """处理聊天命令。

        支持: /status, /new, /reset, /compact, /help, /stats
        """
        cmd = command.strip().lower()

        if cmd in ("/status", "/state"):
            status = await self.get_status()
            return f"会话数: {status['sessions']['active']}\n模型: {status['llm']['model']}\n工具数: {status['tools']['total_tools']}"

        elif cmd == "/new":
            session = await self.sessions.create()
            return f"新会话已创建: {session.session_id}"

        elif cmd == "/reset":
            await self.sessions.reset(session_id)
            return "会话已重置"

        elif cmd == "/compact":
            await self.compact(session_id)
            return "上下文已压缩"

        elif cmd == "/stats":
            stats = await self.memory.get_stats()
            mem_usage = stats["frozen"]["usage"]
            return (
                f"记忆用量: {mem_usage['memory']['pct']}% ({mem_usage['memory']['chars']}/{mem_usage['memory']['max']} chars)\n"
                f"技能数: {stats['procedural']['skill_count']}\n"
                f"对话记录: {stats['indexed']['total_conversations']}"
            )

        elif cmd in ("/help", "/?"):
            return (
                "可用命令:\n"
                "/status  - 查看当前状态\n"
                "/new     - 创建新会话\n"
                "/reset   - 重置当前会话\n"
                "/compact - 压缩对话上下文\n"
                "/stats   - 查看记忆统计\n"
                "/help    - 显示此帮助"
            )

        else:
            return f"未知命令: {command}\n输入 /help 查看可用命令"
