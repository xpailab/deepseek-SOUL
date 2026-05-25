"""技能自动生成器 — 从执行追踪中提取可复用模式。

任务完成 → 分析执行追踪 → 识别成功模式 → 生成 SKILL.md。
"""

from __future__ import annotations

import time
from typing import Any

from soul.types import Skill, SkillMeta, SkillType, Trajectory, TrajectoryStep


class SkillGenerator:
    """技能自动生成器。

    从成功执行的轨迹中提取模式，自动生成可复用技能文件。
    """

    def __init__(self, min_steps: int = 2, min_success_rate: float = 0.7):
        self.min_steps = min_steps
        self.min_success_rate = min_success_rate

    async def analyze_trajectory(self, trajectory: Trajectory) -> dict[str, Any]:
        """分析执行轨迹，提取关键信息。"""
        if not trajectory.success or len(trajectory.steps) < self.min_steps:
            return {"generatable": False, "reason": "轨迹太短或未成功"}

        # 提取步骤模式
        pattern = self._extract_pattern(trajectory.steps)

        # 提取使用的工具
        tools_used = self._extract_tools(trajectory.steps)

        # 提取关键决策点
        decisions = self._extract_decisions(trajectory.steps)

        return {
            "generatable": True,
            "task": trajectory.task,
            "pattern": pattern,
            "tools_used": tools_used,
            "decisions": decisions,
            "step_count": len(trajectory.steps),
            "total_tokens": trajectory.total_tokens,
        }

    async def generate(
        self,
        trajectory: Trajectory,
        custom_name: str = "",
        custom_description: str = "",
    ) -> Skill | None:
        """从轨迹生成技能。"""
        analysis = await self.analyze_trajectory(trajectory)
        if not analysis["generatable"]:
            return None

        # 生成技能名称
        name = custom_name or self._generate_name(trajectory.task)
        description = custom_description or f"自动生成的技能: {trajectory.task[:150]}"

        # 生成 SKILL.md 内容
        content = self._generate_content(name, description, analysis)

        # 提取触发词
        triggers = self._extract_triggers(trajectory.task, analysis)

        meta = SkillMeta(
            name=name,
            version="1.0.0",
            description=description,
            type=SkillType.EVOLVED,
            triggers=triggers,
            gepa_generation=1,
            fitness_score=0.5,  # 初始适应度
        )

        return Skill(meta=meta, content=content)

    async def generate_batch(
        self,
        trajectories: list[Trajectory],
        min_success: int = 3,
    ) -> list[Skill]:
        """从多条轨迹批量生成技能。

        只有同一任务成功多次才生成技能。
        """
        # 按任务分组
        task_groups: dict[str, list[Trajectory]] = {}
        for t in trajectories:
            if t.success:
                key = self._normalize_task(t.task)
                task_groups.setdefault(key, []).append(t)

        skills: list[Skill] = []
        for task, trajs in task_groups.items():
            if len(trajs) >= min_success:
                # 使用最佳轨迹
                best = sorted(trajs, key=lambda x: x.total_tokens)[0]
                skill = await self.generate(best)
                if skill:
                    skills.append(skill)

        return skills

    def _extract_pattern(self, steps: list[TrajectoryStep]) -> list[dict[str, Any]]:
        """提取执行模式。"""
        pattern: list[dict[str, Any]] = []
        for step in steps:
            p: dict[str, Any] = {
                "step": step.step_index,
                "role": step.role.value,
                "has_tools": bool(step.tool_calls),
                "tools": [tc.name for tc in step.tool_calls],
            }
            if step.content:
                p["summary"] = step.content[:100]
            pattern.append(p)
        return pattern

    def _extract_tools(self, steps: list[TrajectoryStep]) -> list[str]:
        """提取使用的工具列表。"""
        tools: set[str] = set()
        for step in steps:
            for tc in step.tool_calls:
                tools.add(tc.name)
        return sorted(tools)

    def _extract_decisions(self, steps: list[TrajectoryStep]) -> list[str]:
        """提取关键决策点。"""
        decisions: list[str] = []
        for step in steps:
            if step.tool_calls:
                for tc in step.tool_calls:
                    decisions.append(f"调用 {tc.name}")
            elif step.role.value == "assistant" and len(step.content) > 50:
                decisions.append(step.content[:150] + "...")
        return decisions

    def _generate_name(self, task: str) -> str:
        """从任务描述生成技能名称。"""
        # 取前几个有意义的词
        words = task.lower().strip().split()
        meaningful = [w for w in words if len(w) > 2 and w not in
                      ("the", "how", "what", "when", "where", "为什么", "怎么", "什么")]
        name = "_".join(meaningful[:4])
        return name or f"skill_{int(time.time()) % 10000}"

    def _generate_content(
        self, name: str, description: str, analysis: dict[str, Any]
    ) -> str:
        """生成 SKILL.md 内容。"""
        lines = [
            f"# {name}",
            "",
            "## 描述",
            description,
            "",
            "## 执行步骤",
        ]

        for step in analysis.get("pattern", []):
            tools_str = f" [工具: {', '.join(step['tools'])}]" if step["tools"] else ""
            lines.append(f"{step['step']}. [{step['role']}]{tools_str}")
            if "summary" in step:
                lines.append(f"   {step['summary']}")

        lines.extend([
            "",
            "## 使用的工具",
            ", ".join(analysis.get("tools_used", [])),
            "",
            "## 质量指标",
            f"- 步骤数: {analysis.get('step_count', 0)}",
            f"- Token 消耗: {analysis.get('total_tokens', 0)}",
            "",
            "## 注意",
            "- 此技能由 AI 自动生成（GEPA 第 1 代）",
            "- 随着使用会持续优化进化",
        ])

        return "\n".join(lines)

    def _extract_triggers(self, task: str, analysis: dict[str, Any]) -> list[str]:
        """提取触发关键词。"""
        triggers: list[str] = []

        # 从任务中提取
        tech_keywords = [
            "docker", "git", "python", "node", "react", "api", "database", "deploy",
            "test", "build", "install", "config", "backup", "debug", "log", "monitor",
            "部署", "测试", "安装", "配置", "备份", "调试", "监控", "创建", "更新",
        ]

        task_lower = task.lower()
        for kw in tech_keywords:
            if kw in task_lower:
                triggers.append(kw)

        # 从工具使用中提取
        for tool in analysis.get("tools_used", []):
            triggers.append(tool)

        return triggers[:7]

    @staticmethod
    def _normalize_task(task: str) -> str:
        """标准化任务描述用于分组。"""
        return task.lower().strip()[:100]
