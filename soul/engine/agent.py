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
from soul.llm.registry import AdapterRegistry
from soul.memory.manager import MemoryManager
from soul.prompt.builder import PromptBuilder
from soul.prompt.compressor import ContextCompressor
from soul.tools.builtin.bash import BashTool
from soul.tools.builtin.file import FileTool
from soul.tools.builtin.web import WebTool
from soul.tools.classifier import ResultClassifier
from soul.tools.guardrails import ToolGuardrails
from soul.tools.registry import ToolRegistry
from soul.tools.retry import RetryManager
from soul.types import (
    AgentEvent,
    AgentState,
    LLMConfig,
    Message,
    MessageRole,
    QueueMode,
    SessionState,
    SOULConfig,
    StreamChunk,
    ToolCall,
    ToolResult,
)


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

        # 如果不是直接入队，等待队列处理
        if result not in ("steered",):
            item = await self.lane_queue.dequeue(session_id)
            if item is None:
                return "系统繁忙，请稍后重试"

        try:
            # 更新状态
            await self.sessions.update_state(session_id, AgentState.THINKING)

            # 获取会话历史
            history = await self.sessions.get_history(session_id)

            # 构建系统提示
            if not system_prompt:
                memory_context = await self.memory.query_for_prompt(user_message)
                system_prompt = self.prompt_builder.build_system_prompt(
                    tools=tools or self.tools.to_api_schemas(),
                    extra_context=memory_context,
                )

            # 构建消息
            user_msg = Message(role=MessageRole.USER, content=user_message)
            await self.sessions.add_message(session_id, user_msg)

            full_messages = self.prompt_builder.build_messages(
                history + [user_msg],
                system_prompt=system_prompt,
                tools=tools or self.tools.to_api_schemas(),
            )

            # LLM 推理
            await self.sessions.update_state(session_id, AgentState.EXECUTING)

            config = self.config.llm
            response = await self.llm.chat(
                full_messages,
                tools=tools or self.tools.to_api_schemas(),
                system_prompt=system_prompt,
                config=config,
                provider=config.provider,
            )

            # 处理工具调用
            tool_results: list[ToolResult] = []
            for tc in response.tool_calls:
                tr = await self._execute_tool(tc, session)
                tool_results.append(tr)

            # 保存助手回复
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
            )
            await self.sessions.add_message(session_id, assistant_msg)

            # 更新记忆
            await self.memory.observe_action(user_message)
            await self.memory.store_conversation(session_id, [user_msg, assistant_msg])

            # 更新状态
            await self.sessions.update_state(session_id, AgentState.IDLE)

            return response.content

        finally:
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

        if result not in ("steered",):
            item = await self.lane_queue.dequeue(session_id)
            if item is None:
                yield StreamChunk(content="系统繁忙，请稍后重试", finish_reason="error")
                return

        try:
            await self.sessions.update_state(session_id, AgentState.STREAMING)

            history = await self.sessions.get_history(session_id)

            if not system_prompt:
                memory_context = await self.memory.query_for_prompt(user_message)
                system_prompt = self.prompt_builder.build_system_prompt(
                    tools=tools or self.tools.to_api_schemas(),
                    extra_context=memory_context,
                )

            user_msg = Message(role=MessageRole.USER, content=user_message)
            await self.sessions.add_message(session_id, user_msg)

            full_messages = self.prompt_builder.build_messages(
                history + [user_msg],
                system_prompt=system_prompt,
                tools=tools or self.tools.to_api_schemas(),
            )

            config = self.config.llm
            full_content = ""
            tool_calls: list[ToolCall] = []

            # 注册 steer 回调
            async def steer_cb(text: str) -> None:
                """Steer 注入回调。"""
                pass  # 生产环境实现实际的流注入

            self.lane_queue.register_steer_callback(session_id, steer_cb)

            async for chunk in self.llm.chat_stream(
                full_messages,
                tools=tools or self.tools.to_api_schemas(),
                system_prompt=system_prompt,
                config=config,
                provider=config.provider,
            ):
                if chunk.content:
                    full_content += chunk.content
                if chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                yield chunk

            # 处理工具调用
            tool_results: list[ToolResult] = []
            for tc in tool_calls:
                tr = await self._execute_tool(tc, session)
                tool_results.append(tr)
                if tr.success:
                    yield StreamChunk(content=f"\n[工具 {tc.name} 执行完成]\n")

            # 保存助手回复
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=full_content,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            await self.sessions.add_message(session_id, assistant_msg)

            await self.memory.observe_action(user_message)
            await self.memory.store_conversation(session_id, [user_msg, assistant_msg])

            await self.sessions.update_state(session_id, AgentState.IDLE)

        finally:
            self.lane_queue.mark_done(session_id)

    async def _execute_tool(
        self, tool_call: ToolCall, session: SessionState
    ) -> ToolResult:
        """执行工具调用（带安全检查）。"""
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
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=f"安全检查失败: {reason}",
                classification="denied",
            )

        # 高风险操作需确认
        if tool_def.requires_approval:
            # 在交互模式中应提示用户
            pass

        # 执行
        start = time.time()
        result, error, retries = await self.retry_mgr.execute_with_retry(
            tool_def.handler,
            **tool_call.arguments,
            tool_name=tool_call.name,
        )
        elapsed = (time.time() - start) * 1000

        tool_def.call_count += 1
        if error:
            tool_def.error_count += 1

        return self.classifier.classify(
            tool_call.name,
            result,
            error=error,
            duration_ms=elapsed,
            timeout_seconds=tool_def.timeout_seconds,
        )

    def _register_builtin_tools(self) -> None:
        """注册内置工具。"""
        self.tools.register(BashTool.to_tool_def())
        self.tools.register(FileTool.to_tool_def())
        self.tools.register(WebTool.to_tool_def())

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
            for state in self._all_sessions():
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
        await self.sessions.close_all()
        await self.memory.close()
        await self.llm.close_all()

    def _all_sessions(self) -> list[SessionState]:
        """获取所有会话状态。"""
        return list(self.sessions._sessions.values())

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
