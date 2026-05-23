"""任务阶段管理器 — 复杂任务拆分执行。

将大型任务拆分为多个阶段，每阶段完成后报告进展，
用户确认后继续下一阶段。

使用示例:
    planner = TaskStagePlanner(agent)
    stages = await planner.plan("构建一个完整的Web应用")

    for stage in stages:
        result = await stage.execute()
        if not await confirm_next_stage(result.summary):
            break
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from soul.types import Message, MessageRole


@dataclass
class TaskStage:
    """单个任务阶段。"""

    id: str
    name: str
    description: str
    estimated_tools: int = 10
    dependencies: list[str] = field(default_factory=list)
    completed: bool = False
    result_summary: str = ""
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "estimated_tools": self.estimated_tools,
            "dependencies": self.dependencies,
            "completed": self.completed,
            "result_summary": self.result_summary,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStage:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            estimated_tools=data.get("estimated_tools", 10),
            dependencies=data.get("dependencies", []),
            completed=data.get("completed", False),
            result_summary=data.get("result_summary", ""),
            artifacts=data.get("artifacts", []),
        )


@dataclass
class TaskPlan:
    """任务执行计划。"""

    original_task: str
    stages: list[TaskStage]
    current_stage_index: int = 0
    total_estimated_tools: int = 0

    def __post_init__(self):
        if not self.total_estimated_tools:
            self.total_estimated_tools = sum(s.estimated_tools for s in self.stages)

    def get_current_stage(self) -> TaskStage | None:
        """获取当前阶段。"""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None

    def get_next_stage(self) -> TaskStage | None:
        """获取下一阶段。"""
        next_idx = self.current_stage_index + 1
        if 0 <= next_idx < len(self.stages):
            return self.stages[next_idx]
        return None

    def complete_current_stage(self, summary: str, artifacts: list[str] | None = None):
        """标记当前阶段完成。"""
        stage = self.get_current_stage()
        if stage:
            stage.completed = True
            stage.result_summary = summary
            if artifacts:
                stage.artifacts.extend(artifacts)
        self.current_stage_index += 1

    def is_complete(self) -> bool:
        """检查是否所有阶段都完成。"""
        return self.current_stage_index >= len(self.stages)

    def get_progress_summary(self) -> str:
        """获取进度摘要。"""
        total = len(self.stages)
        completed = sum(1 for s in self.stages if s.completed)
        current = self.get_current_stage()
        current_name = current.name if current else "已完成"
        return f"进度: {completed}/{total} 阶段完成 | 当前: {current_name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_task": self.original_task,
            "stages": [s.to_dict() for s in self.stages],
            "current_stage_index": self.current_stage_index,
            "total_estimated_tools": self.total_estimated_tools,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPlan:
        stages = [TaskStage.from_dict(s) for s in data.get("stages", [])]
        return cls(
            original_task=data["original_task"],
            stages=stages,
            current_stage_index=data.get("current_stage_index", 0),
            total_estimated_tools=data.get("total_estimated_tools", 0),
        )


class TaskStagePlanner:
    """任务阶段规划器。

    分析复杂任务，将其拆分为可管理的阶段。
    """

    PLANNING_PROMPT = """你是一个任务规划专家。请分析用户的复杂任务，将其拆分为多个可执行的阶段。

任务拆分原则:
1. 每个阶段应有明确的目标和交付物
2. 阶段之间应有逻辑依赖关系
3. 每个阶段预计需要5-20次工具调用
4. 优先完成基础架构，再实现功能
5. 每个阶段完成后应可验证

输出格式（JSON）:
{{
    "stages": [
        {{
            "id": "stage_1",
            "name": "阶段名称",
            "description": "详细描述本阶段要做什么",
            "estimated_tools": 15,
            "dependencies": []
        }}
    ],
    "total_estimated_tools": 45
}}

注意:
- 如果任务简单（少于15次工具调用），返回单阶段
- 如果任务复杂，拆分为2-5个阶段
- 确保阶段之间有合理的依赖关系

用户任务: {task}
"""

    def __init__(self, agent):
        """
        Args:
            agent: Agent实例，用于调用LLM
        """
        self.agent = agent

    async def plan(self, task: str) -> TaskPlan:
        """为任务创建执行计划。

        Args:
            task: 用户原始任务描述

        Returns:
            TaskPlan: 任务执行计划
        """
        # 调用LLM进行任务规划
        messages = [
            Message(role=MessageRole.USER, content=self.PLANNING_PROMPT.format(task=task))
        ]

        try:
            response = await self.agent.llm.chat(
                messages,
                config=self.agent.config.llm,
                provider=self.agent.config.llm.provider,
            )

            # 解析JSON响应
            content = response.content.strip()
            # 尝试提取JSON部分
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            stages = []
            for i, stage_data in enumerate(data.get("stages", [])):
                stage = TaskStage(
                    id=stage_data.get("id", f"stage_{i}"),
                    name=stage_data.get("name", f"阶段 {i+1}"),
                    description=stage_data.get("description", ""),
                    estimated_tools=stage_data.get("estimated_tools", 10),
                    dependencies=stage_data.get("dependencies", []),
                )
                stages.append(stage)

            # 如果没有成功解析，创建单阶段计划
            if not stages:
                stages = [TaskStage(
                    id="stage_1",
                    name="完整任务",
                    description=task,
                    estimated_tools=30,
                )]

            return TaskPlan(
                original_task=task,
                stages=stages,
                total_estimated_tools=data.get("total_estimated_tools", sum(s.estimated_tools for s in stages)),
            )

        except Exception as e:
            # 规划失败，创建简单的单阶段计划
            return TaskPlan(
                original_task=task,
                stages=[TaskStage(
                    id="stage_1",
                    name="完整任务",
                    description=task,
                    estimated_tools=50,
                )],
                total_estimated_tools=50,
            )

    def should_use_stages(self, task: str, plan: TaskPlan) -> bool:
        """判断是否应该使用分阶段执行。

        Returns:
            True 如果任务复杂需要分阶段
        """
        # 如果预计工具调用超过阈值，建议分阶段（即使只有1个阶段）
        if plan.total_estimated_tools > 30:
            return True

        # 如果只有一个阶段且预估不大，不需要分阶段
        if len(plan.stages) <= 1:
            return False

        # 任务包含明显的多阶段关键词
        multi_stage_keywords = [
            "项目", "系统", "应用", "平台", "完整", "开发", "构建",
            "project", "build", "system", "application", "develop", "deploy",
        ]
        if any(kw in task.lower() for kw in multi_stage_keywords):
            return True

        return False


def build_stage_prompt(stage: TaskStage, plan: TaskPlan, previous_results: list[str] | None = None) -> str:
    """为阶段执行构建系统提示。

    Args:
        stage: 当前阶段
        plan: 完整计划
        previous_results: 之前阶段的结果摘要

    Returns:
        阶段执行提示
    """
    lines = [
        "## 阶段执行模式",
        "",
        f"**原始任务**: {plan.original_task}",
        f"**当前阶段**: {stage.name}",
        f"**阶段描述**: {stage.description}",
        f"**阶段进度**: {plan.current_stage_index + 1}/{len(plan.stages)}",
        "",
    ]

    if previous_results:
        lines.append("**已完成的工作**:")
        for result in previous_results:
            lines.append(f"  - {result}")
        lines.append("")

    lines.extend([
        "**当前阶段目标**:",
        f"  {stage.description}",
        "",
        "**执行要求**:",
        "  1. 专注于当前阶段的目标",
        "  2. 完成后生成清晰的结果摘要",
        "  3. 列出本阶段交付的文件/成果",
        "  4. 不要开始下一阶段的工作",
        "",
        "阶段完成后，请输出：",
        '  "阶段完成摘要：[简要描述完成的工作]"',
        '  "阶段交付物：[文件列表]"',
    ])

    return "\n".join(lines)


def parse_stage_completion(content: str) -> tuple[str, list[str]]:
    """从Agent回复中解析阶段完成信息。

    Returns:
        (summary, artifacts)
    """
    summary = ""
    artifacts = []

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if "阶段完成摘要" in line or "完成摘要" in line:
            summary = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif "阶段交付物" in line or "交付物" in line:
            artifacts_text = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            artifacts = [a.strip() for a in artifacts_text.replace(",", "\n").split("\n") if a.strip()]

    # 如果没有明确标记，使用最后几行作为摘要
    if not summary and len(lines) > 0:
        summary = lines[-1][:200]

    return summary, artifacts
