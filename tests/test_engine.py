"""引擎系统测试：lane_queue, session, task_stages。"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from soul.types import QueueMode


class TestLaneQueue:
    def test_init(self):
        from soul.engine.lane_queue import LaneQueue
        from soul.types import LaneConfig
        lq = LaneQueue(LaneConfig(max_concurrent=4))
        assert lq is not None
        stats = lq.get_stats()
        assert "global" in stats
        assert "sessions" in stats

    def test_enqueue_dequeue(self):
        from soul.engine.lane_queue import LaneQueue, QueueItem
        from soul.types import LaneConfig
        lq = LaneQueue(LaneConfig(max_concurrent=4))
        item = QueueItem(id="msg_1", session_id="s1", prompt="test", mode=QueueMode.QUEUE)
        assert item.session_id == "s1"
        assert item.prompt == "test"


class TestSessionManager:
    @pytest.mark.asyncio
    async def test_create_session(self):
        tmp = tempfile.mkdtemp(prefix="soul_session_")
        from soul.engine.session import SessionManager
        sm = SessionManager(str(tmp))
        session = await sm.create()
        assert session.session_id is not None
        stats = await sm.list_sessions()
        assert len(stats) >= 1
        shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        tmp = tempfile.mkdtemp(prefix="soul_session_")
        from soul.engine.session import SessionManager
        sm = SessionManager(str(tmp))
        s1 = await sm.get_or_create("main")
        s2 = await sm.get_or_create("main")
        assert s1.session_id == s2.session_id
        shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_add_message_and_history(self):
        tmp = tempfile.mkdtemp(prefix="soul_session_")
        from soul.engine.session import SessionManager
        from soul.types import Message, MessageRole

        sm = SessionManager(str(tmp))
        session = await sm.create(session_key="test_hist")
        msg = Message(role=MessageRole.USER, content="hello")
        await sm.add_message(session.session_id, msg)
        history = await sm.get_history(session.session_id)
        assert len(history) >= 1
        assert history[-1].content == "hello"
        shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_reset(self):
        tmp = tempfile.mkdtemp(prefix="soul_session_")
        from soul.engine.session import SessionManager
        from soul.types import Message, MessageRole

        sm = SessionManager(str(tmp))
        session = await sm.get_or_create("test_reset")
        await sm.add_message(session.session_id, Message(role=MessageRole.USER, content="msg"))
        await sm.reset(session.session_id)
        history = await sm.get_history(session.session_id)
        assert len(history) == 0
        shutil.rmtree(tmp, ignore_errors=True)


class TestTaskStages:
    def test_task_plan_dataclass(self):
        from soul.engine.task_stages import TaskPlan, TaskStage

        plan = TaskPlan(
            original_task="test task",
            stages=[
                TaskStage(id="s1", name="Stage 1", description="desc", estimated_tools=2),
                TaskStage(id="s2", name="Stage 2", description="desc2", estimated_tools=1),
            ],
            total_estimated_tools=3,
        )
        assert len(plan.stages) == 2
        assert not plan.is_complete()
        assert plan.get_current_stage().id == "s1"

        plan.complete_current_stage("done", [])
        assert plan.get_current_stage().id == "s2"
        assert not plan.is_complete()

        plan.complete_current_stage("done", [])
        assert plan.is_complete()

    def test_parse_stage_completion(self):
        from soul.engine.task_stages import parse_stage_completion
        summary, artifacts = parse_stage_completion("阶段完成，成功部署")
        assert isinstance(summary, str)
        assert isinstance(artifacts, list)
