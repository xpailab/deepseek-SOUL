"""工作记忆 + 执行计划 — 会话级推理增强。

提供:
- WorkingMemory: 追踪已尝试方法、排除方向、中间发现、错误模式
- ExecutionPlan: 结构化执行计划，含预期结果和失败回退
- 自纠错上下文: 为 LLM 诊断和修正提供结构化信息

所有功能零额外 LLM 调用——通过 prompt 注入和结果解析实现。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """单个执行步骤。"""
    step: int
    action: str
    tool: str = ""
    expected: str = ""
    fallback: str = ""
    completed: bool = False
    result_summary: str = ""
    success: bool | None = None

    def mark_done(self, success: bool, summary: str = ""):
        self.completed = True
        self.success = success
        self.result_summary = summary[:200]


@dataclass
class ExecutionPlan:
    """任务执行计划。"""
    task: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = 0.0

    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def current_step(self) -> PlanStep | None:
        for s in self.steps:
            if not s.completed:
                return s
        return None

    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.completed)
        return f"{done}/{len(self.steps)}"

    def to_prompt(self) -> str:
        if not self.steps:
            return ""
        lines = ["## 执行计划"]
        for s in self.steps:
            status = "✓" if s.success else ("✗" if s.success is False else "·")
            lines.append(f"  {status} 步骤{s.step}: {s.action}")
            if s.expected:
                lines.append(f"     预期: {s.expected}")
            if s.fallback:
                lines.append(f"     失败时: {s.fallback}")
            if s.result_summary:
                lines.append(f"     结果: {s.result_summary}")
        return "\n".join(lines)

    @classmethod
    def parse_from_text(cls, text: str, task: str = "") -> ExecutionPlan:
        """从 LLM 响应中解析 JSON 计划块。"""
        plan = cls(task=task, created_at=time.time())

        # 尝试提取 ```json ... ``` 块
        json_match = re.search(
            r'```(?:json)?\s*\n?\s*(\[\s*\{.*?\}\s*\])',
            text, re.DOTALL | re.IGNORECASE
        )
        if not json_match:
            # 尝试直接提取 JSON 数组
            json_match = re.search(
                r'\[\s*\{.*?"step".*?\}\s*\]',
                text, re.DOTALL
            )
        if not json_match:
            # 尝试解析 [计划] 文本格式
            return cls._parse_text_plan(text, task)

        try:
            raw = json_match.group(1)
            steps_data = json.loads(raw)
            for i, s in enumerate(steps_data):
                plan.steps.append(PlanStep(
                    step=s.get("step", i + 1),
                    action=s.get("action", s.get("description", "")),
                    tool=s.get("tool", ""),
                    expected=s.get("expected", s.get("expected_result", "")),
                    fallback=s.get("fallback", s.get("alternative", "")),
                ))
        except (json.JSONDecodeError, KeyError):
            return cls._parse_text_plan(text, task)

        return plan

    @classmethod
    def _parse_text_plan(cls, text: str, task: str = "") -> ExecutionPlan:
        """从自然语言文本中提取步骤（数字编号列表）。"""
        plan = cls(task=task, created_at=time.time())
        # 匹配 "1. xxx" 或 "步骤1: xxx" 格式
        pattern = r'(?:^|\n)\s*(?:\d+[\.\)、]|步骤\s*\d+[：:]) \s*(.+?)(?=\n\s*(?:\d+[\.\)、]|步骤\s*\d+[：:])|$)'
        matches = re.findall(pattern, text, re.MULTILINE)
        for i, m in enumerate(matches):
            action = m.strip()[:200]
            plan.steps.append(PlanStep(step=i + 1, action=action))
        return plan


@dataclass
class WorkingMemory:
    """会话级工作记忆。

    追踪当前任务中的:
    - 已尝试的方法和结果
    - 已排除的方向
    - 中间发现和洞察
    - 错误模式（避免重复犯错）
    - 当前执行计划
    """

    attempts: list[dict[str, Any]] = field(default_factory=list)
    ruled_out: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    error_patterns: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    code_writes: list[str] = field(default_factory=list)  # 本轮写入的代码文件路径
    project_files: dict[str, str] = field(default_factory=dict)  # 已创建的文件→接口摘要
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    _diagnosis_count: int = 0
    _prompt_dirty: bool = True  # 缓存控制：内容变更时置 True

    # --- 记录方法 ---

    def record_attempt(self, action: str, tool: str = "", result: str = "", success: bool = False):
        self._prompt_dirty = True
        self.attempts.append({
            "action": action,
            "tool": tool,
            "result": result[:300],
            "success": success,
            "time": time.time(),
        })

    def record_error(self, tool: str, error: str, diagnosis: str = "", fix: str = ""):
        self._prompt_dirty = True
        self.error_patterns.append({
            "tool": tool,
            "error": error[:300],
            "diagnosis": diagnosis[:300],
            "fix": fix[:300],
            "time": time.time(),
        })

    def record_verification(self, tool: str, passed: bool, issues: list[str] | None = None, suggestions: list[str] | None = None):
        self._prompt_dirty = True
        self.verifications.append({
            "tool": tool,
            "passed": passed,
            "issues": issues or [],
            "suggestions": suggestions or [],
            "time": time.time(),
        })

    def rule_out(self, direction: str):
        if direction not in self.ruled_out:
            self._prompt_dirty = True
            self.ruled_out.append(direction)

    def add_finding(self, finding: str):
        if finding not in self.findings:
            self._prompt_dirty = True
            self.findings.append(finding)

    def set_plan(self, plan: ExecutionPlan):
        self._prompt_dirty = True
        self.execution_plan = plan

    def record_project_file(self, filepath: str, content_preview: str = ""):
        """记录已创建的项目文件及其接口摘要——防止跨轮次 API 漂移。"""
        # 提取关键接口信息（头文件中的 class/function 声明）
        import re
        summary = content_preview[:500] if content_preview else ""
        if not summary and filepath.endswith((".h", ".hpp")):
            # 只存前几行关键声明
            summary = f"[头文件] {filepath}"
        elif filepath.endswith(".cpp"):
            summary = f"[源文件] {filepath}"
        self.project_files[filepath] = summary
        self._prompt_dirty = True

    def get_project_context(self) -> str:
        """生成项目文件清单——让 Agent 记住自己创建了什么。"""
        if not self.project_files:
            return ""
        lines = ["## 已创建的项目文件（注意 API 一致性）"]
        # 头文件优先
        headers = [(fp, s) for fp, s in self.project_files.items() if fp.endswith((".h", ".hpp"))]
        sources = [(fp, s) for fp, s in self.project_files.items() if not fp.endswith((".h", ".hpp"))]
        for fp, summary in headers[-10:] + sources[-10:]:
            lines.append(f"  - {fp}")
        return "\n".join(lines)

    def record_edit_failure(self, filepath: str, error: str = ""):
        """追踪文件编辑失败次数——检测逐行修补死循环。"""
        # 简化：直接追踪 error_patterns 中连续同文件的编辑失败
        pass

    def needs_full_rewrite(self) -> str:
        """检测是否陷入逐行修补死循环。
        条件：最近工具执行中有"编辑文件+运行失败"的循环，同一文件≥2次。
        返回：需要重写的文件路径，空字符串表示不需要。
        """
        recent = self.attempts[-8:]
        if len(recent) < 2:
            return ""

        # 检测模式: file编辑 → bash运行失败 → file再编辑
        code_exts = (".py", ".js", ".ts", ".go", ".rs", ".java", ".sh")
        file_edit_count: dict[str, int] = {}
        run_failures_after_edit = 0

        for i, a in enumerate(recent):
            tool = a.get("tool", "")
            result_str = str(a.get("result", ""))

            # 文件编辑
            if tool in ("write_file", "write", "edit_file", "edit", "file"):
                for line in result_str.split("\n"):
                    for ext in code_exts:
                        if ext in line:
                            fp = line.strip().rsplit(ext, 1)[0] + ext
                            fp = fp.split()[-1] if fp.split() else fp
                            file_edit_count[fp] = file_edit_count.get(fp, 0) + 1

            # bash 运行（且失败）
            if tool in ("bash", "shell", "exec", "run") and not a.get("success"):
                # 检查前面是否有文件编辑
                prev_edits = any(
                    recent[j].get("tool", "") in ("write_file", "write", "edit_file", "edit", "file")
                    for j in range(max(0, i - 2), i)
                )
                if prev_edits:
                    run_failures_after_edit += 1

        # 有 ≥2 次"编辑后运行失败" → 触发重写
        if run_failures_after_edit >= 2 and file_edit_count:
            fp = max(file_edit_count, key=file_edit_count.get)
            if file_edit_count[fp] >= 2:
                return fp
        return ""

    def get_rewrite_prompt(self, filepath: str) -> str:
        """生成全文重写指令。"""
        return (
            f"\n## ⚠️ 逐行修补死循环检测 —— 立即切换策略\n"
            f"文件 `{filepath}` 已经多次编辑仍然失败。\n"
            f"**不要再逐行修补了。** 直接使用 write_file 完整重写整个文件。\n"
            f"把修复了所有已知错误的完整版本一次性写入。"
        )

    # --- 查询方法 ---

    def has_tried(self, action_fragment: str) -> bool:
        return any(action_fragment in a["action"] for a in self.attempts)

    def last_error(self) -> dict[str, Any] | None:
        return self.error_patterns[-1] if self.error_patterns else None

    def repeated_failures(self, threshold: int = 2) -> bool:
        """检查最近 N 次尝试是否都失败。"""
        recent = self.attempts[-threshold:]
        return len(recent) == threshold and all(not a["success"] for a in recent)

    # --- Prompt 生成 ---

    def to_prompt(self) -> str:
        """生成可注入 system prompt 的工作记忆上下文。"""
        sections = []

        # 执行计划进度
        plan_text = self.execution_plan.to_prompt()
        if plan_text:
            sections.append(plan_text)

        # 最近的尝试
        if self.attempts:
            recent = self.attempts[-6:]
            lines = ["## 已尝试的方法"]
            for a in recent:
                status = "✓" if a["success"] else "✗"
                tool_info = f" [{a['tool']}]" if a["tool"] else ""
                lines.append(f"  {status}{tool_info} {a['action']}: {a['result'][:120]}")
            sections.append("\n".join(lines))

        # 已排除的方向
        if self.ruled_out:
            lines = ["## 已排除的方向（不要重复尝试）"]
            for r in self.ruled_out[-5:]:
                lines.append(f"  - ✗ {r}")
            sections.append("\n".join(lines))

        # 中间发现
        if self.findings:
            lines = ["## 中间发现（可用于后续步骤）"]
            for f in self.findings[-5:]:
                lines.append(f"  - {f}")
            sections.append("\n".join(lines))

        # 验证失败
        failed_verifications = [v for v in self.verifications[-5:] if not v["passed"]]
        if failed_verifications:
            lines = ["## 输出验证失败（结果不符合预期）"]
            for v in failed_verifications:
                lines.append(f"  - ✗ {v['tool']}: {'; '.join(v['issues'][:3])}")
                for s in v["suggestions"][:2]:
                    lines.append(f"    → {s}")
            sections.append("\n".join(lines))

        # 错误模式
        if self.error_patterns:
            recent_errors = self.error_patterns[-3:]
            lines = ["## 最近的错误和修复方案"]
            for e in recent_errors:
                lines.append(f"  - 工具 {e['tool']} 出错: {e['error'][:120]}")
                if e["diagnosis"]:
                    lines.append(f"    诊断: {e['diagnosis'][:150]}")
                if e["fix"]:
                    lines.append(f"    修正: {e['fix'][:150]}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else ""

    def get_correction_prompt(self) -> str:
        """生成自纠错提示词——当多次失败时注入。"""
        if not self.error_patterns:
            return ""

        last = self.error_patterns[-1]
        lines = [
            "\n[自纠错提示]",
            f"上一步 {last['tool']} 执行失败: {last['error'][:200]}",
        ]
        if last["diagnosis"]:
            lines.append(f"原因分析: {last['diagnosis'][:200]}")
        if last["fix"]:
            lines.append(f"建议修复: {last['fix'][:200]}")

        if self.repeated_failures(2):
            lines.append("⚠️ 连续多次失败，请尝试完全不同的方法，而非继续修正当前方案。")
            if self.ruled_out:
                lines.append(f"已排除: {', '.join(self.ruled_out[-3:])}")

        return "\n".join(lines)

    def get_planning_prompt(self, task: str) -> str:
        """生成执行计划提示词。"""
        return f"""在开始执行之前，请先制定一个简短的执行计划。用以下 JSON 格式输出：

```json
[
  {{"step": 1, "action": "具体做什么", "tool": "工具名", "expected": "预期结果", "fallback": "失败时的替代方案"}},
  ...
]
```

任务: {task}

然后立即开始执行第一步。"""

    def clear(self):
        """清空工作记忆（新任务开始）。"""
        self.attempts.clear()
        self.ruled_out.clear()
        self.findings.clear()
        self.error_patterns.clear()
        self.verifications.clear()
        self.code_writes.clear()
        self.project_files.clear()
        self.execution_plan = ExecutionPlan()
        self._diagnosis_count = 0
        self._prompt_dirty = True
