"""GEPA 进化引擎 — Genetic-Pareto Prompt Evolution.

基于 Hermes Agent 的 GEPA (ICLR 2026 Oral) 重新实现，针对 DeepSoul 优化。

核心流程:
    初始化 (加载当前版本为基线)
        ↓
    变异 (分析执行追踪 → 找根本原因 → 生成针对性文本变体)
        ↓
    评估 (并行运行所有变体)
        ↓
    选择 (帕累托最优 — 准确率 × 成本 × 延迟)
        ↓
    迭代 (保留最优个体继续进化)

特点:
- 纯 API 调用，无需 GPU
- 每次优化成本 $2-10
- 仅需 3 个示例
- 优化提示文本（非模型权重）
- 多目标帕累托优化
"""

from __future__ import annotations

import asyncio
import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from soul.types import Skill, SkillMeta


@dataclass
class GEPAConfig:
    """GEPA 进化配置。"""
    population_size: int = 8       # 种群大小
    max_iterations: int = 10       # 最大迭代次数
    mutation_rate: float = 0.3     # 变异率
    crossover_rate: float = 0.5    # 交叉率
    elite_count: int = 2           # 精英保留数
    max_skill_size_kb: int = 15    # 技能文件大小上限
    tool_desc_max_chars: int = 500 # 工具描述上限
    test_count: int = 3            # 最少测试用例数
    early_stop_fitness: float = 0.95  # 早停适应度阈值
    parallel_eval: bool = True     # 并行评估
    verbose: bool = False          # 详细日志


@dataclass
class FitnessScore:
    """适应度分数 — 多目标帕累托优化。"""
    accuracy: float = 0.0    # 准确率（最大化）
    cost: float = 0.0        # API token 消耗（最小化 → 用 1/cost）
    latency: float = 0.0     # 延迟 ms（最小化 → 用 1/latency）
    overall: float = 0.0     # 综合分数

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "cost": round(self.cost, 4),
            "latency": round(self.latency, 4),
            "overall": round(self.overall, 4),
        }


@dataclass
class Individual:
    """种群个体 — 一个技能变体。"""
    skill: Skill
    fitness: FitnessScore = field(default_factory=FitnessScore)
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)


class GEPAEngine:
    """GEPA 进化引擎。

    使用遗传算法 + 帕累托多目标优化来进化提示文本。

    使用示例:
        engine = GEPAEngine(evaluator_func=my_eval)
        improved_skill = await engine.evolve(skill, test_cases)
    """

    def __init__(
        self,
        evaluator_func: Callable[..., Any] | None = None,
        config: GEPAConfig | None = None,
    ):
        """
        Args:
            evaluator_func: 评估函数 async (skill, test_cases) -> FitnessScore
            config: 进化配置
        """
        self.evaluator = evaluator_func or self._default_evaluator
        self.config = config or GEPAConfig()
        self._history: list[dict[str, Any]] = []

    async def evolve(
        self,
        skill: Skill,
        test_cases: list[dict[str, Any]],
        execution_traces: list[dict[str, Any]] | None = None,
    ) -> Skill:
        """进化一个技能。

        Args:
            skill: 要进化的技能
            test_cases: 测试用例 [{"input": ..., "expected": ...}, ...]
            execution_traces: 之前的执行追踪（用于变异分析）

        Returns:
            进化后的技能（帕累托最优）
        """
        if len(test_cases) < self.config.test_count:
            return skill  # 测试用例不足，跳过进化

        # 1. 初始化种群
        population = await self._initialize_population(skill)

        best_individual = population[0]
        best_individual.fitness = await self._evaluate(skill, test_cases)
        best_individual.generation = 0

        if self.config.verbose:
            print(f"GEPA 初始化: 基线适应度 {best_individual.fitness.overall:.4f}")

        # 2. 进化循环
        for gen in range(1, self.config.max_iterations + 1):
            # 变异
            offspring = await self._mutate_population(
                population, execution_traces or [], test_cases
            )

            # 交叉
            offspring = await self._crossover_population(offspring)

            # 评估
            for ind in offspring:
                ind.fitness = await self._evaluate(ind.skill, test_cases)
                ind.generation = gen

            # 合并并选择
            combined = population + offspring
            population = self._select_pareto(combined, self.config.population_size)

            # 更新最佳个体
            gen_best = max(population, key=lambda x: x.fitness.overall)
            if gen_best.fitness.overall > best_individual.fitness.overall:
                best_individual = gen_best

            self._history.append({
                "generation": gen,
                "best_fitness": best_individual.fitness.to_dict(),
                "population_avg": sum(
                    i.fitness.overall for i in population
                ) / len(population),
            })

            if self.config.verbose:
                print(
                    f"  第 {gen} 代: 最佳={best_individual.fitness.overall:.4f}, "
                    f"准确率={best_individual.fitness.accuracy:.4f}"
                )

            # 早停
            if best_individual.fitness.overall >= self.config.early_stop_fitness:
                if self.config.verbose:
                    print(f"  早停: 已达目标适应度 {self.config.early_stop_fitness}")
                break

        # 3. 更新技能
        evolved = best_individual.skill
        evolved.meta.gepa_generation = best_individual.generation
        evolved.meta.fitness_score = best_individual.fitness.overall
        # 版本号迭代
        parts = evolved.meta.version.split(".")
        evolved.meta.version = f"{parts[0]}.{parts[1]}.{int(parts[2]) + best_individual.generation}"
        evolved.updated_at = time.time()

        return evolved

    async def _initialize_population(self, skill: Skill) -> list[Individual]:
        """初始化种群。"""
        population = [Individual(skill=copy.deepcopy(skill))]

        # 生成变体
        for i in range(self.config.population_size - 1):
            variant = copy.deepcopy(skill)
            variant.meta.name = f"{skill.meta.name}_v{i}"
            # 随机微调内容
            variant.content = self._apply_random_variation(variant.content)
            population.append(Individual(skill=variant))

        return population

    async def _mutate_population(
        self,
        population: list[Individual],
        traces: list[dict[str, Any]],
        test_cases: list[dict[str, Any]],
    ) -> list[Individual]:
        """变异操作 — 基于执行追踪分析生成改进变体。"""
        offspring: list[Individual] = []

        for ind in population:
            if random.random() < self.config.mutation_rate:
                variant = copy.deepcopy(ind.skill)

                # 分析失败模式并针对性改进
                improvements = self._analyze_failures(traces, variant.content)

                if improvements:
                    variant.content = self._apply_improvements(
                        variant.content, improvements
                    )
                else:
                    # 随机变异
                    variant.content = self._apply_random_variation(variant.content)

                variant.meta.name = f"{ind.skill.meta.name}_m{random.randint(0, 999)}"

                offspring.append(Individual(
                    skill=variant,
                    parent_ids=[ind.skill.meta.name],
                ))

        return offspring

    async def _crossover_population(
        self, offspring: list[Individual]
    ) -> list[Individual]:
        """交叉操作 — 合并两个父代的优点。"""
        if len(offspring) < 2:
            return offspring

        crossed: list[Individual] = []
        random.shuffle(offspring)

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.config.crossover_rate:
                p1 = offspring[i].skill
                p2 = offspring[i + 1].skill

                # 在段落边界处交叉
                child = copy.deepcopy(p1)
                p1_sections = p1.content.split("\n\n")
                p2_sections = p2.content.split("\n\n")

                if len(p1_sections) > 1 and len(p2_sections) > 1:
                    crossover_point = random.randint(1, min(len(p1_sections), len(p2_sections)) - 1)
                    child.content = "\n\n".join(
                        p1_sections[:crossover_point] + p2_sections[crossover_point:]
                    )

                child.meta.name = f"{p1.meta.name}_x{p2.meta.name[-4:]}"
                crossed.append(Individual(
                    skill=child,
                    parent_ids=[p1.meta.name, p2.meta.name],
                ))

        return offspring + crossed

    def _select_pareto(
        self, population: list[Individual], k: int
    ) -> list[Individual]:
        """帕累托选择 — 多目标优化选择。

        保留:
        1. 精英个体（总体分数 top-k）
        2. 帕累托前沿个体（各维度最优）
        """
        # 按总体分数排序
        population.sort(key=lambda x: x.fitness.overall, reverse=True)

        selected = population[:self.config.elite_count]

        # 添加各维度的最优个体
        if len(selected) < k:
            best_accuracy = max(population, key=lambda x: x.fitness.accuracy)
            if best_accuracy not in selected:
                selected.append(best_accuracy)

        if len(selected) < k:
            best_cost = min(population, key=lambda x: x.fitness.cost)
            if best_cost not in selected:
                selected.append(best_cost)

        if len(selected) < k:
            best_latency = min(population, key=lambda x: x.fitness.latency)
            if best_latency not in selected:
                selected.append(best_latency)

        # 填充剩余位置
        for ind in population:
            if len(selected) >= k:
                break
            if ind not in selected:
                selected.append(ind)

        return selected[:k]

    async def _evaluate(
        self, skill: Skill, test_cases: list[dict[str, Any]]
    ) -> FitnessScore:
        """评估技能适应度。"""
        return await self.evaluator(skill, test_cases)

    async def _default_evaluator(
        self, skill: Skill, test_cases: list[dict[str, Any]]
    ) -> FitnessScore:
        """默认评估器 — 基于规则的简单评分。"""
        # 准确率: 基于内容质量和完整性
        accuracy = self._score_content_quality(skill.content)

        # 成本: 基于文本长度（越短越好）
        cost = len(skill.content) / 5000  # 标准化
        cost_score = max(0, 1 - cost)

        # 延迟: 基于步骤复杂度
        step_count = skill.content.count("\n- ") + skill.content.count("\n1. ")
        latency = max(0, 1 - step_count / 20)

        overall = 0.5 * accuracy + 0.25 * cost_score + 0.25 * latency

        return FitnessScore(
            accuracy=accuracy,
            cost=cost_score,
            latency=latency,
            overall=overall,
        )

    def _score_content_quality(self, content: str) -> float:
        """评分内容质量。"""
        score = 0.5  # 基础分

        # 有描述
        if "## 描述" in content or "## Description" in content:
            score += 0.1

        # 有步骤
        if "## 执行步骤" in content or "## Steps" in content:
            score += 0.1

        # 有触发条件
        if "## 触发条件" in content or "## Triggers" in content:
            score += 0.1

        # 有注意事项
        if "## 注意" in content or "## Notes" in content:
            score += 0.1

        # 长度适中
        length = len(content)
        if 300 < length < 10000:
            score += 0.1

        # 结构清晰（有编号列表）
        if content.count("\n1. ") >= 2:
            score += 0.1

        return min(1.0, score)

    def _analyze_failures(
        self,
        traces: list[dict[str, Any]],
        current_content: str,
    ) -> list[str]:
        """分析执行追踪中的失败原因，生成改进建议。"""
        improvements: list[str] = []

        for trace in traces:
            for step in trace.get("steps", []):
                # 检测失败步骤
                if not step.get("success", True):
                    error = step.get("error", "")
                    if "permission" in error.lower():
                        improvements.append("添加权限检查步骤")
                    elif "timeout" in error.lower():
                        improvements.append("添加超时处理和重试逻辑")
                    elif "not found" in error.lower():
                        improvements.append("添加文件/资源存在性检查")
                    else:
                        improvements.append(f"添加错误处理: {error[:100]}")

                # 检测低效操作
                if step.get("duration_ms", 0) > 30000:
                    improvements.append("优化长时间操作，添加进度检查点")

        return improvements[:5]

    def _apply_improvements(self, content: str, improvements: list[str]) -> str:
        """应用改进建议到技能内容。"""
        if not improvements:
            return content

        lines = content.split("\n")
        insert_pos = len(lines)

        # 在注意事项之前插入
        for i, line in enumerate(lines):
            if line.startswith("## 注意") or line.startswith("## Notes"):
                insert_pos = i
                break

        # 插入改进
        improvement_lines = ["", "## 改进建议 (GEPA 自动生成)"]
        for imp in improvements:
            improvement_lines.append(f"- {imp}")

        return "\n".join(lines[:insert_pos] + improvement_lines + lines[insert_pos:])

    def _apply_random_variation(self, content: str) -> str:
        """应用随机文本变异。"""
        variations = [
            # 添加更多细节
            lambda c: c + "\n\n## 补充说明\n- 此步骤可能需要根据环境调整",
            # 重组步骤
            lambda c: c.replace("1. ", "步骤 1: "),
            # 添加示例
            lambda c: c + "\n\n## 示例\n```\n# 示例代码待补充\n```",
            # 简化描述
            lambda c: c.replace("## 描述", "## 概述"),
            # 添加检查点
            lambda c: c + "\n\n## 验证\n- 检查所有步骤是否成功完成",
        ]

        variation = random.choice(variations)
        return variation(content)

    def get_history(self) -> list[dict[str, Any]]:
        """获取进化历史。"""
        return self._history
