"""轨迹压缩器 — 将完整执行轨迹压缩到指定 token 预算内。

保留关键决策点和工具调用，去除冗余中间步骤。
输出可直接用于模型微调。
"""

from __future__ import annotations

from typing import Any

from soul.types import Trajectory, TrajectoryStep


class TrajectoryCompressor:
    """轨迹压缩器。

    策略:
    1. 保留第一个和最后一个步骤（任务和结果）
    2. 保留工具调用步骤
    3. 保留关键决策步骤（内容 > 100 chars）
    4. 对中间步骤做滑动窗口截断
    5. LLM 摘要作为最后手段
    """

    def __init__(self, max_tokens: int = 8000, min_keep_steps: int = 3):
        self.max_tokens = max_tokens
        self.min_keep_steps = min_keep_steps

    def compress(self, trajectory: Trajectory) -> Trajectory:
        """压缩单条轨迹。"""
        if len(trajectory.steps) <= self.min_keep_steps:
            return trajectory

        token_count = trajectory.total_tokens
        if token_count <= self.max_tokens:
            return trajectory

        # 标记关键步骤
        for step in trajectory.steps:
            step.metadata["keep"] = self._is_key_step(step)

        # 分步压缩
        kept = self._select_steps(trajectory.steps)

        # 如果仍然超出，截断内容
        compressed_steps = self._truncate_content(kept)

        # 生成摘要
        summary = self._generate_summary(
            [s for s in trajectory.steps if s not in kept]
        )

        return Trajectory(
            id=trajectory.id,
            session_id=trajectory.session_id,
            task=trajectory.task,
            steps=compressed_steps,
            success=trajectory.success,
            total_duration_ms=trajectory.total_duration_ms,
            total_tokens=sum(s.token_count for s in compressed_steps),
            metadata={
                **trajectory.metadata,
                "compressed": True,
                "original_steps": len(trajectory.steps),
                "original_tokens": token_count,
                "summary": summary,
            },
        )

    def compress_batch(
        self, trajectories: list[Trajectory]
    ) -> list[Trajectory]:
        """批量压缩。"""
        return [self.compress(t) for t in trajectories]

    def _is_key_step(self, step: TrajectoryStep) -> bool:
        """判断是否为关键步骤。"""
        # 工具调用是关键
        if step.tool_calls:
            return True
        # 有实质性内容的步骤
        if len(step.content) > 100:
            return True
        # 用户输入是关键
        if step.role.value == "user":
            return True
        return False

    def _select_steps(self, steps: list[TrajectoryStep]) -> list[TrajectoryStep]:
        """选择要保留的步骤。"""
        n = len(steps)
        if n <= self.min_keep_steps:
            return steps

        # 始终保留第一步和最后一步
        keep_indices = {0, n - 1}

        # 保留所有关键步骤
        for i, step in enumerate(steps):
            if step.metadata.get("keep"):
                keep_indices.add(i)

        # 如果还不够，采样中间步骤
        sorted_indices = sorted(keep_indices)
        return [steps[i] for i in sorted_indices]

    def _truncate_content(
        self, steps: list[TrajectoryStep]
    ) -> list[TrajectoryStep]:
        """截断步骤内容。"""
        budget_per_step = self.max_tokens // max(1, len(steps)) * 3  # ~chars per step

        for step in steps:
            if len(step.content) > budget_per_step:
                step.content = step.content[:budget_per_step] + "..."
                step.metadata["truncated"] = True

        return steps

    def _generate_summary(self, removed_steps: list[TrajectoryStep]) -> str:
        """生成被移除步骤的摘要。"""
        if not removed_steps:
            return ""

        parts = []
        for step in removed_steps:
            role = step.role.value
            content = step.content[:100] if step.content else ""
            tools = [tc.name for tc in step.tool_calls]
            tool_str = f" [工具: {', '.join(tools)}]" if tools else ""
            parts.append(f"[{role}]{tool_str} {content}")

        return " | ".join(parts)

    def compress_for_training(self, trajectory: Trajectory) -> dict[str, Any]:
        """压缩为可直接用于训练的格式。"""
        compressed = self.compress(trajectory)

        messages = []
        for step in compressed.steps:
            msg = {"role": step.role.value, "content": step.content}
            if step.tool_calls:
                msg["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in step.tool_calls
                ]
            messages.append(msg)

        return {
            "id": compressed.id,
            "task": compressed.task,
            "success": compressed.success,
            "messages": messages,
        }
