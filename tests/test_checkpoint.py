"""检查点系统测试。"""

from __future__ import annotations

import tempfile

from soul.engine.checkpoint import Checkpoint, CheckpointManager


class TestCheckpoint:
    def test_create(self):
        cp = Checkpoint(
            session_id="sess_001",
            task="deploy app",
            plan_steps=[
                {"step": 1, "action": "build", "completed": True, "success": True, "result_summary": "done"},
                {"step": 2, "action": "push", "completed": False},
            ],
        )
        assert cp.session_id == "sess_001"
        assert cp.completed_steps == 1
        assert cp.total_steps == 2
        assert not cp.is_complete

    def test_complete(self):
        cp = Checkpoint(
            session_id="sess_x",
            task="test",
            plan_steps=[
                {"step": 1, "action": "a", "completed": True, "success": True},
                {"step": 2, "action": "b", "completed": True, "success": True},
            ],
        )
        assert cp.is_complete

    def test_serialization(self):
        cp = Checkpoint(
            session_id="s1",
            task="task1",
            plan_steps=[{"step": 1, "action": "do", "completed": False}],
            findings=["found something"],
            ruled_out=["bad approach"],
        )
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.session_id == "s1"
        assert cp2.findings == ["found something"]
        assert cp2.ruled_out == ["bad approach"]


class TestCheckpointManager:
    def test_save_and_load(self):
        tmp = tempfile.mkdtemp(prefix="soul_cp_")
        mgr = CheckpointManager(str(tmp))

        from soul.engine.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.add_finding("port 8080 in use")
        wm.rule_out("direct rm approach")

        mgr.save(
            session_id="test_sess",
            task="deploy something",
            plan_steps=[
                {"step": 1, "action": "build", "completed": True, "success": True, "result_summary": "ok"},
                {"step": 2, "action": "deploy", "completed": False},
            ],
            working_memory=wm,
        )

        cp = mgr.load("test_sess")
        assert cp is not None
        assert cp.task == "deploy something"
        assert cp.total_steps == 2
        assert cp.completed_steps == 1
        assert "port 8080" in cp.findings[0]
        assert not cp.is_complete

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_list_incomplete(self):
        tmp = tempfile.mkdtemp(prefix="soul_cp_")
        mgr = CheckpointManager(str(tmp))

        # 保存两个检查点：一个未完成、一个完成
        mgr.save(session_id="incomplete_1", task="t1", plan_steps=[
            {"step": 1, "action": "a", "completed": True},
            {"step": 2, "action": "b", "completed": False},
        ])
        mgr.save(session_id="complete_1", task="t2", plan_steps=[
            {"step": 1, "action": "a", "completed": True, "success": True},
        ])

        incomplete = mgr.list_incomplete()
        assert len(incomplete) == 1
        assert incomplete[0].session_id == "incomplete_1"

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_load_latest(self):
        tmp = tempfile.mkdtemp(prefix="soul_cp_")
        mgr = CheckpointManager(str(tmp))

        mgr.save(session_id="older", task="old", plan_steps=[
            {"step": 1, "action": "x", "completed": False},
        ])
        import time
        time.sleep(0.1)
        mgr.save(session_id="newer", task="new", plan_steps=[
            {"step": 1, "action": "y", "completed": False},
        ])

        latest = mgr.load_latest()
        assert latest is not None
        assert latest.session_id == "newer"

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_mark_complete(self):
        tmp = tempfile.mkdtemp(prefix="soul_cp_")
        mgr = CheckpointManager(str(tmp))

        mgr.save(session_id="to_complete", task="t", plan_steps=[
            {"step": 1, "action": "a", "completed": False},
        ])
        assert mgr.load("to_complete") is not None

        mgr.mark_complete("to_complete")
        assert mgr.load("to_complete") is None

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_resume_context(self):
        tmp = tempfile.mkdtemp(prefix="soul_cp_")
        mgr = CheckpointManager(str(tmp))

        from soul.engine.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.add_finding("registry seems slow")
        wm.rule_out("direct SSH deploy")

        mgr.save(
            session_id="resume_test",
            task="deploy to production",
            plan_steps=[
                {"step": 1, "action": "build image", "completed": True, "success": True, "result_summary": "built ok"},
                {"step": 2, "action": "push to registry", "completed": True, "success": False, "result_summary": "timeout"},
                {"step": 3, "action": "deploy to k8s", "completed": False},
            ],
            working_memory=wm,
        )

        cp = mgr.load("resume_test")
        ctx = mgr.get_resume_context(cp)
        assert "断点续跑" in ctx
        assert "deploy to production" in ctx
        assert "2/3" in ctx or "已完成" in ctx
        assert "下一步" in ctx
        assert "registry seems slow" in ctx
        assert "direct SSH deploy" in ctx

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
