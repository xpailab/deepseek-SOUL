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
from typing import Any, AsyncIterator

from soul.config.manager import ConfigManager
from soul.engine.lane_queue import LaneQueue, QueueItem
from soul.engine.session import SessionManager
from soul.engine.task_stages import (
    TaskStagePlanner,
    TaskPlan,
    build_stage_prompt,
    parse_stage_completion,
)
from soul.engine.working_memory import WorkingMemory, ExecutionPlan
from soul.llm.registry import AdapterRegistry
from soul.memory.manager import MemoryManager
from soul.prompt.builder import PromptBuilder
from soul.prompt.compressor import ContextCompressor
from soul.tools.builtin.bash import BashTool
from soul.tools.builtin.file import FileTool
from soul.tools.builtin.web import WebTool
from soul.tools.builtin.browser import BrowserTool
from soul.tools.builtin.windows import WindowsTool
from soul.tools.classifier import ResultClassifier
from soul.tools.guardrails import ToolGuardrails
from soul.tools.registry import ToolRegistry
from soul.tools.retry import RetryManager
from soul.types import (
    AgentEvent,
    AgentState,
    LLMConfig,
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

        # 工作记忆 + 执行计划（会话级推理增强）
        self.working_memory = WorkingMemory()

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
            # 更新状态
            await self.sessions.update_state(session_id, AgentState.THINKING)

            # 获取会话历史
            history = await self.sessions.get_history(session_id)

            # 构建系统提示（含技能匹配 + 记忆检索）
            memory_context = await self.memory.query_for_prompt(user_message)
            matched_skills = self.memory.procedural.match(user_message, top_k=2)
            base_system_prompt = self.prompt_builder.build_system_prompt(
                matched_skills=matched_skills,
                tools=tools or self.tools.to_api_schemas(),
                extra_context=memory_context,
            )
            if system_prompt:
                base_system_prompt = base_system_prompt + "\n\n" + system_prompt

            # 工作记忆增强 + 执行规划注入
            enhanced_prompt = self._build_enhanced_prompt(
                base_system_prompt, user_message, first_round=True
            )
            self.working_memory.clear()

            # 构建消息
            user_msg = Message(role=MessageRole.USER, content=user_message)

            full_messages = self.prompt_builder.build_messages(
                history + [user_msg],
                system_prompt=enhanced_prompt,
                tools=tools or self.tools.to_api_schemas(),
            )

            # LLM 推理 + 工具调用循环
            max_rounds = 50
            max_total_tools = 100
            consecutive_fails = 0
            actual_rounds = 0
            config = self.config.llm
            all_tool_results: list[ToolResult] = []
            final_content = ""
            current_messages = list(full_messages)
            saved_len = len(current_messages)

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

            # 循环结束：生成任务执行报告
            report = _build_task_report(
                user_message, actual_rounds, all_tool_results,
                len(all_tool_results) >= max_total_tools,
                consecutive_fails >= 5,
            )
            final_content = final_content + "\n\n" + report if final_content else report

            # 只保存本轮新产生的消息
            for msg in current_messages[saved_len:]:
                await self.sessions.add_message(session_id, msg)

            if not final_content:
                final_content = "抱歉，任务未能完成。请重试或简化您的请求。"

            await self.memory.observe_action(user_message)
            await self.memory.store_conversation(session_id, current_messages)
            summary = f"用户: {user_message[:100]} | 回复: {final_content[:100]}"
            await self.memory.remember(summary, layer=MemoryLayer.FROZEN)

            # 技能自动学习 — 成功任务 → 提取模式 → 生成/进化技能
            if self.config.skill.auto_generate and consecutive_fails < 5:
                await self._learn_from_task(user_message, current_messages, all_tool_results)

            await self.sessions.update_state(session_id, AgentState.IDLE)
            return final_content

        except Exception as e:
            # 异常时确保返回错误信息而不是空
            error_msg = f"执行任务时出错: {str(e)}"
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
            await self.sessions.update_state(session_id, AgentState.STREAMING)

            history = await self.sessions.get_history(session_id)

            # 构建基础prompt，工作记忆增强
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
                base_system_prompt, user_message, first_round=True
            )
            self.working_memory.clear()

            user_msg = Message(role=MessageRole.USER, content=user_message)

            full_messages = self.prompt_builder.build_messages(
                history + [user_msg],
                system_prompt=enhanced_prompt,
                tools=tools or self.tools.to_api_schemas(),
            )

            config = self.config.llm
            full_content = ""

            # 注册 steer 回调
            steer_queue: asyncio.Queue[str] = asyncio.Queue()

            async def steer_cb(text: str) -> None:
                await steer_queue.put(text)

            self.lane_queue.register_steer_callback(session_id, steer_cb)

            # LLM 推理 + 工具调用循环
            max_rounds = 50
            max_total_tools = 100
            consecutive_fails = 0
            actual_rounds = 0
            current_messages = list(full_messages)
            saved_len = len(current_messages)
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

            # 生成任务执行报告
            report = _build_task_report(
                user_message, actual_rounds, all_tool_results,
                len(all_tool_results) >= max_total_tools,
                consecutive_fails >= 5,
            )
            report_chunk = "\n\n" + report
            yield StreamChunk(content=report_chunk)
            full_content = full_content + report_chunk if full_content else report

            # 只保存本轮新产生的消息
            for msg in current_messages[saved_len:]:
                await self.sessions.add_message(session_id, msg)

            if not full_content:
                full_content = "抱歉，任务未能完成。请重试或简化您的请求。"
                yield StreamChunk(content=full_content)

            await self.memory.observe_action(user_message)
            await self.memory.store_conversation(session_id, current_messages)
            summary = f"用户: {user_message[:100]} | 回复: {full_content[:100]}"
            await self.memory.remember(summary, layer=MemoryLayer.FROZEN)

            # 技能自动学习
            if self.config.skill.auto_generate and consecutive_fails < 5:
                await self._learn_from_task(user_message, current_messages, all_tool_results)

            await self.sessions.update_state(session_id, AgentState.IDLE)

        except Exception as e:
            # 流式异常时发送错误标记
            yield StreamChunk(content=f"\n[执行出错: {str(e)}]", finish_reason="error")

        finally:
            self.lane_queue.untrack_active(session_id)
            if acquired:
                self.lane_queue.mark_done(session_id)

    # ═══════════════════════════════════════════
    # 问题解决增强：计划 + 工作记忆 + 自纠错 + 验证
    # ═══════════════════════════════════════════

    def _build_enhanced_prompt(
        self, base_prompt: str, user_message: str, first_round: bool
    ) -> str:
        """构建增强系统提示——注入工作记忆和执行计划指令。"""
        wm = self.working_memory
        parts = [base_prompt]

        # 注入工作记忆上下文
        wm_text = wm.to_prompt()
        if wm_text:
            parts.append(wm_text)

        # 首轮：注入规划指令
        if first_round and wm.execution_plan.is_empty():
            parts.append(wm.get_planning_prompt(user_message))

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

        # 记录每步工具执行结果
        for tr in round_results:
            if hasattr(tr, 'success'):
                wm.record_attempt(
                    action=f"{tr.name}",
                    tool=tr.name,
                    result=str(tr.result)[:200] if tr.result else "",
                    success=tr.success,
                )
                if not tr.success and tr.error:
                    wm.record_error(
                        tool=tr.name,
                        error=tr.error,
                    )

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
            retries = 0

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
