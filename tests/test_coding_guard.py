"""编码自动化行为测试：强制验证 + 小步快跑 + 回归检查。"""

from __future__ import annotations

import pytest

from soul.engine.agent import Agent


class TestCodingTaskDetection:
    def test_coding_task(self):
        assert Agent._is_coding_task("写一个FastAPI用户认证模块") is True
        assert Agent._is_coding_task("创建 models/user.py") is True
        assert Agent._is_coding_task("重构 soul/memory/manager.py") is True
        assert Agent._is_coding_task("开发 REST API 接口") is True
        assert Agent._is_coding_task("实现一个缓存装饰器") is True
        assert Agent._is_coding_task("修改 main.go 的端口配置") is True
        assert Agent._is_coding_task("加一个日志中间件") is True
        assert Agent._is_coding_task("写个 .js 脚本") is True

    def test_non_coding_task(self):
        assert Agent._is_coding_task("你好") is False
        assert Agent._is_coding_task("今天天气怎么样") is False
        assert Agent._is_coding_task("列出当前目录文件") is False
        assert Agent._is_coding_task("帮我查一下这个报错是什么意思") is False


class TestCodingCadence:
    def test_generates_for_coding_task(self):
        agent = Agent.__new__(Agent)
        prompt = agent._coding_cadence_prompt("写一个REST API接口")
        assert "小步快跑" in prompt
        assert "验证" in prompt

    def test_empty_for_non_coding(self):
        agent = Agent.__new__(Agent)
        prompt = agent._coding_cadence_prompt("你好")
        assert prompt == ""


class TestBuildVerifyPrompt:
    def test_python(self):
        prompt = Agent._build_verify_prompt("write_file", "app/main.py")
        assert "py_compile" in prompt or "python" in prompt
        assert "app/main.py" in prompt

    def test_go(self):
        prompt = Agent._build_verify_prompt("write_file", "main.go")
        assert "go build" in prompt or "go vet" in prompt

    def test_js(self):
        prompt = Agent._build_verify_prompt("write_file", "index.js")
        assert "node --check" in prompt

    def test_non_code(self):
        assert Agent._build_verify_prompt("write_file", "README.md") == ""
        assert Agent._build_verify_prompt("write_file", "") == ""


class TestCodingGuard:
    def test_empty_when_no_code(self):
        from soul.engine.working_memory import WorkingMemory
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        guard = agent._coding_guard_from_memory()
        assert guard == ""

    def test_generates_when_code_written(self):
        from soul.engine.working_memory import WorkingMemory
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        agent.working_memory.code_writes.append("app/main.py")
        guard = agent._coding_guard_from_memory()
        assert "立即验证" in guard or "py_compile" in guard or "python" in guard
        # 使用后应清空
        assert len(agent.working_memory.code_writes) == 0

    def test_multiple_files(self):
        from soul.engine.working_memory import WorkingMemory
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        agent.working_memory.code_writes.append("app/main.py")
        agent.working_memory.code_writes.append("app/models.py")
        agent.working_memory.code_writes.append("app/routes.py")
        guard = agent._coding_guard_from_memory()
        assert guard != ""


class TestRegressionGuard:
    def test_empty_when_no_plan(self):
        from soul.engine.working_memory import WorkingMemory
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        guard = agent._regression_guard()
        assert guard == ""

    def test_not_triggered_early(self):
        from soul.engine.working_memory import WorkingMemory, ExecutionPlan, PlanStep
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        plan = ExecutionPlan(task="test")
        plan.steps = [
            PlanStep(step=1, action="a"),
            PlanStep(step=2, action="b"),
            PlanStep(step=3, action="c"),
            PlanStep(step=4, action="d"),
            PlanStep(step=5, action="e"),
        ]
        plan.steps[0].mark_done(True, "ok")
        agent.working_memory.execution_plan = plan
        guard = agent._regression_guard()
        assert guard == ""  # 1/5 < 80%

    def test_triggered_near_completion(self):
        from soul.engine.working_memory import WorkingMemory, ExecutionPlan, PlanStep
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        plan = ExecutionPlan(task="test")
        plan.steps = [
            PlanStep(step=1, action="a"),
            PlanStep(step=2, action="b"),
            PlanStep(step=3, action="c"),
            PlanStep(step=4, action="d"),
            PlanStep(step=5, action="e"),
        ]
        for s in plan.steps[:4]:
            s.mark_done(True, "ok")
        agent.working_memory.execution_plan = plan
        guard = agent._regression_guard()
        assert "回归" in guard
        assert "测试" in guard or "test" in guard.lower()

    def test_triggered_only_once(self):
        from soul.engine.working_memory import WorkingMemory, ExecutionPlan, PlanStep
        agent = Agent.__new__(Agent)
        agent.working_memory = WorkingMemory()
        plan = ExecutionPlan(task="test")
        plan.steps = [
            PlanStep(step=1, action="a"),
            PlanStep(step=2, action="b"),
            PlanStep(step=3, action="c"),
            PlanStep(step=4, action="d"),
        ]
        for s in plan.steps[:3]:
            s.mark_done(True, "ok")
        agent.working_memory.execution_plan = plan
        # 触发一次 (3/4 >= 80%)
        guard1 = agent._regression_guard()
        assert guard1 != ""
        # 不再触发（has_tried 记录存在）
        guard2 = agent._regression_guard()
        assert guard2 == ""
