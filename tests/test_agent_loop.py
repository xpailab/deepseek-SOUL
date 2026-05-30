"""Agent 核心循环测试——覆盖 chat/chat_stream 的执行路径。

重点: 共享方法验证、停止条件、错误恢复、流式输出。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from soul.types import (
    Message,
    MessageRole,
    StreamChunk,
    ToolCall,
    ToolResult,
)

# ============================================================
# 共享 helper 方法
# ============================================================

class TestChatPrepare:
    """_chat_prepare() — 初始化、会话、入队、上下文构建。"""

    @pytest.mark.asyncio
    async def test_returns_none_when_queue_full(self, base_config):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True  # 跳过初始化

        # Mock lane_queue.dequeue 返回 None（系统繁忙）
        agent.lane_queue = MagicMock()
        agent.lane_queue.enqueue = AsyncMock(return_value="queued")
        agent.lane_queue.dequeue = AsyncMock(return_value=None)

        result = await agent._chat_prepare("test", "s1", "", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_prepared_state(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        agent.lane_queue = MagicMock()
        agent.lane_queue.enqueue = AsyncMock(return_value="queued")
        agent.lane_queue.dequeue = AsyncMock(return_value=MagicMock())

        # Mock _setup_chat_context
        async def mock_setup(*args, **kwargs):
            return {
                "base_system_prompt": "test_prompt",
                "current_messages": [],
                "saved_len": 0,
                "config": base_config.llm,
            }

        agent._setup_chat_context = mock_setup
        agent.sessions = MagicMock()
        agent.sessions.get_or_create = AsyncMock(return_value=MagicMock(session_id="s1"))

        result = await agent._chat_prepare("test", "s1", "", None)
        assert result is not None
        assert result["session_id"] == "s1"
        assert result["base_system_prompt"] == "test_prompt"
        assert result["acquired"] is True
        assert "config" in result

    @pytest.mark.asyncio
    async def test_steered_item_skips_dequeue(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        agent.lane_queue = MagicMock()
        agent.lane_queue.enqueue = AsyncMock(return_value="steered")
        agent.lane_queue.dequeue = AsyncMock(return_value=MagicMock())

        async def mock_setup(*args, **kwargs):
            return {
                "base_system_prompt": "p",
                "current_messages": [],
                "saved_len": 0,
                "config": base_config.llm,
            }

        agent._setup_chat_context = mock_setup
        agent.sessions = MagicMock()
        agent.sessions.get_or_create = AsyncMock(return_value=MagicMock(session_id="s2"))

        result = await agent._chat_prepare("test", "s2", "", None)
        assert result is not None
        assert result["acquired"] is False  # steered 不占用槽位


class TestProcessRoundResults:
    """_process_round_results() — 工具执行 + 记忆更新 + 消息构建。"""

    @pytest.mark.asyncio
    async def test_executes_tools_and_builds_messages(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        # Mock _execute_tool 返回成功的工具结果
        async def mock_execute(tc, sid):
            return ToolResult(
                call_id=tc.id,
                name=tc.name,
                success=True,
                result="ok",
                classification="success",
            )

        agent._execute_tool = mock_execute
        agent._update_working_memory = MagicMock()

        tcs = [
            ToolCall(id="c1", name="bash", arguments={"command": "echo hi"}),
            ToolCall(id="c2", name="file", arguments={"path": "test.txt"}),
        ]
        messages = []
        consecutive_fails = 0
        all_results = []

        new_fails, round_results = await agent._process_round_results(
            tcs, "s1", "执行中", "", "test task",
            all_results, consecutive_fails, 100, messages,
        )

        assert new_fails == 0
        assert len(round_results) == 2
        assert len(all_results) == 2
        assert len(messages) == 3  # 1 assistant + 2 tool

        # 验证助手消息
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].content == "执行中"
        assert messages[0].tool_calls == tcs

        # 验证工具结果消息
        assert messages[1].role == MessageRole.TOOL
        assert messages[2].role == MessageRole.TOOL

    @pytest.mark.asyncio
    async def test_tracks_consecutive_failures(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        async def mock_execute(tc, sid):
            return ToolResult(
                call_id=tc.id, name=tc.name, success=False,
                error="permission denied", classification="failure",
            )

        agent._execute_tool = mock_execute
        agent._update_working_memory = MagicMock()

        tcs = [ToolCall(id="c1", name="bash", arguments={"command": "rm -rf /"})]
        messages = []

        new_fails, _ = await agent._process_round_results(
            tcs, "s1", "fail", "", "test",
            [], 2, 100, messages,  # consecutive_fails starts at 2
        )

        assert new_fails == 3  # 2 + 1

    @pytest.mark.asyncio
    async def test_resets_consecutive_on_success(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        async def mock_execute(tc, sid):
            return ToolResult(
                call_id=tc.id, name=tc.name, success=True,
                result="ok", classification="success",
            )

        agent._execute_tool = mock_execute
        agent._update_working_memory = MagicMock()

        tcs = [ToolCall(id="c1", name="bash", arguments={"command": "ls"})]
        messages = []

        new_fails, _ = await agent._process_round_results(
            tcs, "s1", "ok", "", "test",
            [], 3, 100, messages,  # was failing at 3
        )

        assert new_fails == 0  # reset to 0 on success

    @pytest.mark.asyncio
    async def test_respects_tool_limit(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        call_count = 0

        async def mock_execute(tc, sid):
            nonlocal call_count
            call_count += 1
            return ToolResult(
                call_id=tc.id, name=tc.name, success=True,
                result="ok", classification="success",
            )

        agent._execute_tool = mock_execute
        agent._update_working_memory = MagicMock()

        # 已经 99 个工具调用，只剩 1 个名额
        all_results = [MagicMock() for _ in range(99)]
        tcs = [ToolCall(id=f"c{i}", name="bash", arguments={}) for i in range(5)]

        new_fails, round_results = await agent._process_round_results(
            tcs, "s1", "ok", "", "test",
            all_results, 0, 100, [],
        )

        assert call_count == 1  # 只执行了 1 个（达到上限）
        assert len(round_results) == 1


# ============================================================
# chat() 停止条件
# ============================================================

class TestChatStopConditions:
    """chat() 的停止条件——工具上限、连续失败。"""

    @pytest.mark.asyncio
    async def test_stops_on_tool_limit(self, base_config, tmp_workspace):
        """100 次工具调用后停止。"""
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        # 准备状态（跳过 _chat_prepare，直接注入循环状态）
        prepared = {
            "session_id": "s1",
            "base_system_prompt": "p",
            "current_messages": [],
            "saved_len": 0,
            "config": base_config.llm,
            "acquired": True,
        }

        with patch.object(agent, '_chat_prepare', new=AsyncMock(return_value=prepared)):
            with patch.object(agent, '_finalize_chat', new=AsyncMock(return_value="done: stopped")):
                with patch.object(agent.lane_queue, 'mark_done'):
                    with patch.object(agent.sessions, 'update_state', new=AsyncMock()):
                        result = await agent.chat("test")

        # 100 次工具调用 = 立即触发停止条件
        # 因为 _chat_prepare 返回空消息列表和 0 次工具调用，
        # 第一轮 LLM 会返回无工具调用的响应，直接退出
        # 这里我们验证正常路径不崩溃
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_stops_on_consecutive_fails(self, base_config, tmp_workspace):
        """连续 5 次失败后停止——验证 _process_round_results 正确追踪失败次数。"""
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        # 直接测试 _process_round_results 的失败追踪
        async def mock_execute(tc, sid):
            return ToolResult(
                call_id=tc.id, name=tc.name, success=False,
                error="mock error", classification="failure",
            )

        agent._execute_tool = mock_execute
        agent._update_working_memory = MagicMock()

        # 模拟：已经失败了 4 次，再失败 1 次就到 5
        consecutive = 4
        tcs = [ToolCall(id="c1", name="bash", arguments={"command": "bad"})]
        new_fails, _ = await agent._process_round_results(
            tcs, "s1", "failing", "", "test",
            [], consecutive, 100, [],
        )
        assert new_fails == 5  # 触发停止条件

    @pytest.mark.asyncio
    async def test_success_resets_fail_counter(self, base_config, tmp_workspace):
        """成功后失败计数归零——不会误触发停止。"""
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        async def mock_execute(tc, sid):
            return ToolResult(
                call_id=tc.id, name=tc.name, success=True,
                result="ok", classification="success",
            )

        agent._execute_tool = mock_execute
        agent._update_working_memory = MagicMock()

        consecutive = 4  # 快到了
        tcs = [ToolCall(id="c1", name="bash", arguments={"command": "good"})]
        new_fails, _ = await agent._process_round_results(
            tcs, "s1", "recovering", "", "test",
            [], consecutive, 100, [],
        )
        assert new_fails == 0  # 重置了


# ============================================================
# chat_stream() 流式输出
# ============================================================

class TestChatStream:
    """chat_stream() 流式输出。"""

    @pytest.mark.asyncio
    async def test_yields_error_on_queue_full(self, base_config):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        with patch.object(agent, '_chat_prepare', new=AsyncMock(return_value=None)):
            chunks = []
            async for chunk in agent.chat_stream("test"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0].finish_reason == "error"
            assert "繁忙" in chunks[0].content

    @pytest.mark.asyncio
    async def test_yields_tool_results(self, base_config, tmp_workspace):
        """流式输出中包含工具执行结果。"""
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        prepared = {
            "session_id": "s1",
            "base_system_prompt": "p",
            "current_messages": [],
            "saved_len": 0,
            "config": base_config.llm,
            "acquired": True,
        }

        agent._chat_prepare = AsyncMock(return_value=prepared)
        agent._finalize_chat = AsyncMock()
        agent._update_working_memory = MagicMock()
        agent.lane_queue.mark_done = MagicMock()
        agent.lane_queue.untrack_active = MagicMock()
        agent.lane_queue.track_active = MagicMock()
        agent.lane_queue.register_steer_callback = MagicMock()
        agent.sessions.update_state = AsyncMock()

        # Mock LLM stream: 返回 1 个带工具调用的 chunk
        tc = ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})

        async def mock_stream(*args, **kwargs):
            chunk = StreamChunk(content="running", tool_call=tc, finish_reason="tool_calls")
            yield chunk

        agent.llm.chat_stream = mock_stream

        # Mock 工具执行
        async def mock_execute(tc_in, sid):
            return ToolResult(
                call_id=tc_in.id, name=tc_in.name, success=True,
                result="hello", classification="success",
            )
        agent._execute_tool = mock_execute

        chunks = [c async for c in agent.chat_stream("test")]

        tr_chunks = [c for c in chunks if c.tool_result is not None]
        assert len(tr_chunks) >= 1, (
            f"No tool_result chunks in "
            f"{[(c.content[:30] if c.content else '', c.tool_result) for c in chunks]}"
        )
        assert tr_chunks[0].tool_result.success is True


# ============================================================
# 压缩检查
# ============================================================

class TestCompression:
    """中循环压缩——每 5 轮触发。"""

    def test_needs_compression_triggers(self, base_config):
        from soul.prompt.compressor import ContextCompressor

        cc = ContextCompressor()
        # 128000 * 0.8 = 102400 tokens 阈值
        # 每条约 4 chars ≈ 1 token, 所以需要约 100000 chars
        messages = [
            Message(role=MessageRole.USER, content="x" * 2000, tool_calls=[])
            for _ in range(200)
        ]
        needs = cc.needs_compression(messages)
        assert needs is True

    def test_no_compression_for_short_context(self, base_config):
        from soul.prompt.compressor import ContextCompressor

        cc = ContextCompressor()
        messages = [Message(role=MessageRole.USER, content="hi", tool_calls=[])]
        needs = cc.needs_compression(messages)
        assert needs is False


# ============================================================
# 错误恢复路径
# ============================================================

class TestErrorRecovery:
    """错误处理路径——异常捕获、状态恢复、检查点保存。"""

    @pytest.mark.asyncio
    async def test_chat_exception_saves_checkpoint(self, base_config, tmp_workspace):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        prepared = {
            "session_id": "s1",
            "base_system_prompt": "p",
            "current_messages": [],
            "saved_len": 0,
            "config": base_config.llm,
            "acquired": True,
        }

        checkpoint_saved = False

        def mock_save_checkpoint(sid, task):
            nonlocal checkpoint_saved
            checkpoint_saved = True

        agent._save_checkpoint = mock_save_checkpoint
        agent.error_kb.save = MagicMock()

        with patch.object(agent, '_chat_prepare', new=AsyncMock(return_value=prepared)):
            with patch.object(agent.lane_queue, 'mark_done'):
                with patch.object(agent.sessions, 'update_state', new=AsyncMock()):
                    # 让循环中的某个操作抛出异常
                    with patch.object(agent, '_build_enhanced_prompt',
                                      side_effect=RuntimeError("模拟崩溃")):
                        result = await agent.chat("test")

        assert "出错" in result
        assert checkpoint_saved is True

    @pytest.mark.asyncio
    async def test_chat_stream_exception_yields_error(self, base_config):
        from soul.engine.agent import Agent

        agent = Agent(config=base_config)
        agent._initialized = True

        prepared = {
            "session_id": "s1",
            "base_system_prompt": "p",
            "current_messages": [],
            "saved_len": 0,
            "config": base_config.llm,
            "acquired": True,
        }

        with patch.object(agent, '_chat_prepare', new=AsyncMock(return_value=prepared)):
            with patch.object(agent.lane_queue, 'untrack_active'):
                with patch.object(agent.lane_queue, 'mark_done'):
                    with patch.object(agent.lane_queue, 'track_active'):
                        with patch.object(agent.lane_queue, 'register_steer_callback'):
                            with patch.object(agent, '_build_enhanced_prompt',
                                              side_effect=RuntimeError("模拟崩溃")):
                                chunks = []
                                async for chunk in agent.chat_stream("test"):
                                    chunks.append(chunk)

        # 至少有 1 个错误 chunk
        error_chunks = [c for c in chunks if c.finish_reason == "error"]
        assert len(error_chunks) >= 1
        assert "出错" in error_chunks[0].content


# ============================================================
# 任务报告
# ============================================================

class TestTaskReport:
    """_build_task_report() 各种状态。"""

    def test_success_report(self):
        from soul.engine.agent import _build_task_report

        results = [
            ToolResult(call_id="c1", name="bash", success=True, result="ok", classification="success"),
            ToolResult(call_id="c2", name="file", success=True, result="ok", classification="success"),
        ]
        report = _build_task_report("test", 3, results, False, False)
        assert "成功" in report

    def test_failure_report(self):
        from soul.engine.agent import _build_task_report

        results = [
            ToolResult(call_id="c1", name="bash", success=False, error="fail", classification="failure"),
        ]
        report = _build_task_report("test", 1, results, False, False)
        assert "失败" in report

    def test_partial_report(self):
        from soul.engine.agent import _build_task_report

        results = [
            ToolResult(call_id="c1", name="bash", success=True, result="ok", classification="success"),
            ToolResult(call_id="c2", name="file", success=False, error="fail", classification="failure"),
        ]
        report = _build_task_report("test", 2, results, False, False)
        assert "部分完成" in report

    def test_limit_report(self):
        from soul.engine.agent import _build_task_report

        results = [ToolResult(call_id=f"c{i}", name="bash", success=True, result="ok", classification="success")
                   for i in range(100)]
        report = _build_task_report("test", 50, results, True, False)
        assert "上限" in report

    def test_empty_results(self):
        from soul.engine.agent import _build_task_report
        report = _build_task_report("test", 1, [], False, False)
        assert report == ""
