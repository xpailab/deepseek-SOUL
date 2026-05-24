"""工作记忆 + 执行计划模块测试。"""

from __future__ import annotations

import pytest

from soul.engine.working_memory import ExecutionPlan, PlanStep, WorkingMemory


class TestPlanStep:
    def test_create(self):
        step = PlanStep(step=1, action="创建目录", tool="bash", expected="目录创建成功", fallback="使用 file 工具")
        assert step.step == 1
        assert step.completed is False
        assert step.success is None

    def test_mark_done(self):
        step = PlanStep(step=1, action="创建文件")
        step.mark_done(True, "文件创建成功")
        assert step.completed is True
        assert step.success is True
        assert "文件创建成功" in step.result_summary

        step2 = PlanStep(step=2, action="删除文件")
        step2.mark_done(False, "权限不足")
        assert step2.success is False


class TestExecutionPlan:
    def test_empty_plan(self):
        plan = ExecutionPlan()
        assert plan.is_empty()
        assert plan.current_step() is None
        assert plan.progress() == "0/0"

    def test_parse_json_plan(self):
        text = '''好的，我来制定计划。
```json
[
  {"step": 1, "action": "创建项目目录", "tool": "bash", "expected": "目录创建成功", "fallback": "使用 mkdir 命令"},
  {"step": 2, "action": "初始化 git 仓库", "tool": "bash", "expected": "git init 成功"},
  {"step": 3, "action": "创建 README.md", "tool": "file", "expected": "文件写入成功"}
]
```
开始执行第一步。'''

        plan = ExecutionPlan.parse_from_text(text, "初始化项目")
        assert not plan.is_empty()
        assert len(plan.steps) == 3
        assert plan.steps[0].action == "创建项目目录"
        assert plan.steps[0].tool == "bash"
        assert plan.steps[0].fallback == "使用 mkdir 命令"

    def test_parse_text_plan(self):
        text = """我来逐步完成：\n1. 首先创建项目目录结构\n2. 然后安装依赖\n3. 最后运行测试"""

        plan = ExecutionPlan.parse_from_text(text, "设置项目")
        assert not plan.is_empty()
        assert len(plan.steps) >= 2  # 文本解析是尽力而为的

    def test_progress_tracking(self):
        plan = ExecutionPlan(task="test")
        plan.steps = [
            PlanStep(step=1, action="step1"),
            PlanStep(step=2, action="step2"),
            PlanStep(step=3, action="step3"),
        ]
        assert plan.progress() == "0/3"
        assert plan.current_step().step == 1

        plan.steps[0].mark_done(True, "ok")
        assert plan.progress() == "1/3"
        assert plan.current_step().step == 2

        plan.steps[1].mark_done(False, "failed")
        assert plan.current_step().step == 3

        plan.steps[2].mark_done(True, "ok")
        assert plan.current_step() is None

    def test_to_prompt(self):
        plan = ExecutionPlan(task="test")
        plan.steps = [PlanStep(step=1, action="创建目录")]
        prompt = plan.to_prompt()
        assert "执行计划" in prompt
        assert "创建目录" in prompt


class TestWorkingMemory:
    def test_record_attempt(self):
        wm = WorkingMemory()
        wm.record_attempt("创建文件", tool="file", result="文件已创建", success=True)
        assert len(wm.attempts) == 1
        assert wm.has_tried("创建文件")

    def test_record_error(self):
        wm = WorkingMemory()
        wm.record_error("bash", "command not found", "命令不存在", "使用正确的命令名")
        assert len(wm.error_patterns) == 1
        assert wm.last_error() is not None

    def test_rule_out(self):
        wm = WorkingMemory()
        wm.rule_out("rm -rf 方案")
        wm.rule_out("rm -rf 方案")  # 不重复
        assert len(wm.ruled_out) == 1

    def test_add_finding(self):
        wm = WorkingMemory()
        wm.add_finding("项目需要 Python 3.11+")
        assert len(wm.findings) == 1

    def test_repeated_failures(self):
        wm = WorkingMemory()
        assert not wm.repeated_failures(2)
        wm.record_attempt("test1", result="fail", success=False)
        wm.record_attempt("test2", result="fail", success=False)
        assert wm.repeated_failures(2)
        wm.record_attempt("test3", result="ok", success=True)
        assert not wm.repeated_failures(2)

    def test_to_prompt_includes_all_sections(self):
        wm = WorkingMemory()
        wm.record_attempt("部署应用", tool="bash", result="docker pushed", success=True)
        wm.record_error("bash", "connection refused", "网络不通", "检查防火墙")
        wm.rule_out("直接 rm -rf /")
        wm.add_finding("端口 8080 已被占用")
        wm.execution_plan = ExecutionPlan(task="deploy")
        wm.execution_plan.steps = [PlanStep(step=1, action="构建镜像")]
        wm.execution_plan.steps[0].mark_done(True, "ok")

        prompt = wm.to_prompt()
        assert "已尝试的方法" in prompt
        assert "已排除的方向" in prompt
        assert "中间发现" in prompt
        assert "最近的错误和修复方案" in prompt
        assert "执行计划" in prompt

    def test_empty_prompt(self):
        wm = WorkingMemory()
        assert wm.to_prompt() == ""

    def test_correction_prompt(self):
        wm = WorkingMemory()
        wm.record_error("bash", "Permission denied: /etc/passwd", "权限不足", "使用用户目录路径")
        correction = wm.get_correction_prompt()
        assert "自纠错提示" in correction
        assert "Permission denied" in correction

    def test_correction_prompt_repeated_failures(self):
        wm = WorkingMemory()
        wm.record_attempt("t1", result="fail", success=False)
        wm.record_error("bash", "error1")
        wm.record_attempt("t2", result="fail", success=False)
        wm.record_error("bash", "error2")
        correction = wm.get_correction_prompt()
        assert "完全不同的方法" in correction or "连续多次失败" in correction

    def test_clear(self):
        wm = WorkingMemory()
        wm.record_attempt("test")
        wm.rule_out("dir a")
        wm.clear()
        assert wm.to_prompt() == ""
        assert len(wm.attempts) == 0
