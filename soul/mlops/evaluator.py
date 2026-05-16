"""LLM-as-Judge 评估器。

用于 GEPA 进化中的技能评估和轨迹质量评分。

评分维度:
- 过程遵循度 (0-1): 是否按技能规定的步骤执行
- 输出正确性 (0-1): 输出是否正确、有用、相关
- 简洁性 (0-1): 是否在 token 预算内完成
"""

from __future__ import annotations

from typing import Any

from soul.types import Message, MessageRole


class LLMJudge:
    """LLM 评判器 — 使用 LLM 评估技能和轨迹质量。"""

    SCORING_PROMPT = """你是一个技能质量评估专家。请根据以下标准评估技能。

评分标准（每项 0-1 分）：
1. 过程遵循度: 技能执行步骤是否清晰、可遵循？步骤是否按正确的顺序排列？
2. 输出正确性: 按此技能执行，输出是否正确、有用、相关？
3. 简洁性: 技能描述是否简洁？是否包含非必要内容？

请对以下技能评分：

{skill_content}

请返回 JSON 格式的评分：
{{"process_adherence": 0.XX, "output_correctness": 0.XX, "conciseness": 0.XX, "overall": 0.XX, "comments": "简要评语"}}
"""

    TRAJECTORY_SCORING_PROMPT = """你是一个轨迹质量评估专家。请评估以下执行轨迹。

任务: {task}

执行步骤:
{steps}

请返回 JSON 格式的评分：
{{"completeness": 0.XX, "efficiency": 0.XX, "correctness": 0.XX, "overall": 0.XX, "comments": "简要评语"}}
"""

    def __init__(self, adapter: Any | None = None):
        """初始化评判器。

        Args:
            adapter: LLM 适配器实例（用于 LLM 评判）
        """
        self.adapter = adapter

    async def evaluate_skill(
        self,
        skill_content: str,
        test_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """评估技能质量。

        Args:
            skill_content: SKILL.md 内容
            test_results: 测试执行结果（可选，用于基于结果的评分）

        Returns:
            评分字典
        """
        if self.adapter:
            return await self._llm_evaluate_skill(skill_content)
        else:
            return self._rule_evaluate_skill(skill_content)

    async def evaluate_trajectory(
        self,
        task: str,
        steps: list[dict[str, Any]],
        success: bool,
    ) -> dict[str, Any]:
        """评估轨迹质量。"""
        if self.adapter:
            return await self._llm_evaluate_trajectory(task, steps)
        else:
            return self._rule_evaluate_trajectory(task, steps, success)

    async def _llm_evaluate_skill(self, skill_content: str) -> dict[str, Any]:
        """使用 LLM 评估技能。"""
        prompt = self.SCORING_PROMPT.format(skill_content=skill_content[:3000])
        msgs = [Message(role=MessageRole.USER, content=prompt)]

        try:
            response = await self.adapter.chat(msgs)
            import json
            # 尝试提取 JSON
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass

        return self._rule_evaluate_skill(skill_content)

    async def _llm_evaluate_trajectory(
        self, task: str, steps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """使用 LLM 评估轨迹。"""
        steps_text = "\n".join(
            f"{i+1}. [{s.get('role', '')}] {str(s.get('content', ''))[:200]}"
            for i, s in enumerate(steps)
        )
        prompt = self.TRAJECTORY_SCORING_PROMPT.format(task=task, steps=steps_text[:3000])
        msgs = [Message(role=MessageRole.USER, content=prompt)]

        try:
            response = await self.adapter.chat(msgs)
            import json
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass

        return self._rule_evaluate_trajectory(task, steps, True)

    @staticmethod
    def _rule_evaluate_skill(skill_content: str) -> dict[str, Any]:
        """基于规则的技能评估。"""
        score = 0.5

        # 结构完整性
        if "## 描述" in skill_content or "## Description" in skill_content:
            score += 0.1
        if "## 执行步骤" in skill_content or "## Steps" in skill_content:
            score += 0.1
        if "## 注意" in skill_content or "## Notes" in skill_content:
            score += 0.05

        # 步骤明确性
        steps_count = skill_content.count("\n- ") + skill_content.count("\n1. ")
        if steps_count >= 3:
            score += 0.1
        elif steps_count >= 1:
            score += 0.05

        # 长度适中
        length = len(skill_content)
        if 300 < length < 10000:
            score += 0.1

        # 简洁性
        if length < 5000:
            score += 0.05

        score = min(1.0, score)

        return {
            "process_adherence": round(score * 0.9, 3),
            "output_correctness": round(score, 3),
            "conciseness": round(1.0 - min(length / 15000, 0.5), 3),
            "overall": round(score, 3),
            "comments": "基于规则的自动评估",
        }

    @staticmethod
    def _rule_evaluate_trajectory(
        task: str, steps: list[dict[str, Any]], success: bool
    ) -> dict[str, Any]:
        """基于规则的轨迹评估。"""
        completeness = 1.0 if success else 0.3
        efficiency = min(1.0, 5.0 / max(1, len(steps)))
        correctness = 1.0 if success else 0.2

        return {
            "completeness": round(completeness, 3),
            "efficiency": round(efficiency, 3),
            "correctness": round(correctness, 3),
            "overall": round((completeness + efficiency + correctness) / 3, 3),
            "comments": "基于规则的自动评估" if success else "任务未成功完成",
        }

    def compare_skills(
        self, skill_a: str, skill_b: str
    ) -> dict[str, Any]:
        """比较两个技能的质量。"""
        result_a = self._rule_evaluate_skill(skill_a)
        result_b = self._rule_evaluate_skill(skill_b)

        winner = "a" if result_a["overall"] > result_b["overall"] else "b"
        if result_a["overall"] == result_b["overall"]:
            winner = "tie"

        return {
            "skill_a_score": result_a["overall"],
            "skill_b_score": result_b["overall"],
            "winner": winner,
            "diff": round(result_a["overall"] - result_b["overall"], 3),
        }
