"""技能注册中心 — 管理技能的生命周期。

支持：
- 技能安装/卸载
- 版本管理
- 依赖解析
- 语义搜索
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from soul.types import Skill, SkillMeta, SkillType


class SkillRegistry:
    """技能注册中心。

    管理三种来源的技能:
    - bundled: 内置捆绑
    - managed: 从注册中心安装
    - workspace: 工作空间自定义
    - evolved: GEPA 自动进化生成
    """

    def __init__(self, skills_dir: str = "~/.soul/skills"):
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self._semantic_index: dict[str, list[str]] = {}  # word -> skill names

    def register(self, skill: Skill) -> None:
        """注册技能。"""
        self._skills[skill.meta.name] = skill
        self._index_skill(skill)

    def unregister(self, name: str) -> bool:
        skill = self._skills.pop(name, None)
        if skill:
            self._remove_from_index(skill)
            # 删除磁盘文件
            filepath = self.skills_dir / f"{name}.skill"
            if filepath.exists():
                filepath.unlink()
            return True
        return False

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def search(
        self,
        query: str = "",
        skill_type: SkillType | None = None,
        tags: list[str] | None = None,
        min_fitness: float = 0.0,
        limit: int = 10,
    ) -> list[Skill]:
        """搜索技能。"""
        results: list[tuple[float, Skill]] = []

        query_lower = query.lower()
        query_words = set(query_lower.split()) if query_lower else set()

        for skill in self._skills.values():
            if skill_type and skill.meta.type != skill_type:
                continue
            if min_fitness and skill.meta.fitness_score < min_fitness:
                continue
            if tags:
                skill_tags = {t.lower() for t in skill.meta.triggers}
                if not skill_tags.intersection({t.lower() for t in tags}):
                    continue

            score = self._score_skill(skill, query_words)
            if score > 0 or not query:
                results.append((score, skill))

        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results[:limit]]

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def get_stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for s in self._skills.values():
            by_type[s.meta.type.value] = by_type.get(s.meta.type.value, 0) + 1
        return {
            "total": len(self._skills),
            "by_type": by_type,
            "avg_fitness": round(
                sum(s.meta.fitness_score for s in self._skills.values()) / max(1, len(self._skills)),
                3,
            ),
        }

    def _index_skill(self, skill: Skill) -> None:
        """建立语义索引。"""
        words = set()
        for text in [skill.meta.name, skill.meta.description, *skill.meta.triggers, skill.content[:500]]:
            for word in text.lower().split():
                word = word.strip(".,;:!?()[]{}\"'")
                if len(word) > 2:
                    words.add(word)

        for word in words:
            self._semantic_index.setdefault(word, []).append(skill.meta.name)

    def _remove_from_index(self, skill: Skill) -> None:
        for names in self._semantic_index.values():
            if skill.meta.name in names:
                names.remove(skill.meta.name)

    def _score_skill(self, skill: Skill, query_words: set[str]) -> float:
        """计算技能与查询的相关性分数。"""
        if not query_words:
            return 0.0

        score = 0.0
        # 名称匹配
        for word in query_words:
            if word in skill.meta.name.lower():
                score += 5.0
            if word in skill.meta.description.lower():
                score += 2.0
            for trigger in skill.meta.triggers:
                if word in trigger.lower():
                    score += 3.0
            # 语义索引匹配
            if word in self._semantic_index and skill.meta.name in self._semantic_index[word]:
                score += 1.0

        # 使用频率和成功率加权
        score *= (0.5 + 0.5 * skill.meta.success_rate)
        score *= (1.0 + min(skill.meta.usage_count / 100, 1.0))
        score *= (1.0 + skill.meta.fitness_score)

        return score
