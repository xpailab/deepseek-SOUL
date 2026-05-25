"""技能加载器 — 从多种来源加载技能。

支持:
- 本地 .skill 文件
- 捆绑技能目录
- 远程注册中心 (ClawHub API)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from soul.types import Skill, SkillMeta, SkillType


class SkillLoader:
    """技能加载器。

    按优先级加载: bundled → managed → workspace → evolved
    """

    def __init__(
        self,
        skills_dir: str = "~/.soul/skills",
        bundled_dir: str | None = None,
    ):
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self.bundled_dir = Path(bundled_dir) if bundled_dir else Path(__file__).parent.parent.parent / "skills" / "bundled"

    async def load_all(self) -> list[Skill]:
        """加载所有来源的技能。"""
        skills: list[Skill] = []

        # 1. 捆绑技能
        if self.bundled_dir.exists():
            for f in self.bundled_dir.glob("*.skill"):
                skill = await self._load_file(f, SkillType.BUNDLED)
                if skill:
                    skills.append(skill)

        # 2. 托管 + 工作空间 + 进化技能
        for f in self.skills_dir.glob("*.skill"):
            skill = await self._load_file(f)
            if skill:
                # 避免重复
                if not any(s.meta.name == skill.meta.name for s in skills):
                    skills.append(skill)

        return skills

    async def load_one(self, name: str) -> Skill | None:
        """加载单个技能。"""
        for d in [self.bundled_dir, self.skills_dir]:
            filepath = d / f"{name}.skill"
            if filepath.exists():
                return await self._load_file(filepath)
        return None

    async def _load_file(
        self, filepath: Path, default_type: SkillType | None = None
    ) -> Skill | None:
        """加载单个技能文件。"""
        try:
            content = filepath.read_text(encoding="utf-8")

            # 解析 YAML frontmatter（可选）
            meta = self._parse_frontmatter(content)
            if default_type and not meta.get("type"):
                meta["type"] = default_type

            # 提取纯文本内容（移除 frontmatter）
            body = self._extract_body(content)

            # 处理 YAML 逗号分隔字符串 → list
            triggers = meta.get("triggers", [])
            if isinstance(triggers, str):
                triggers = [t.strip() for t in triggers.split(",") if t.strip()]

            skill_meta = SkillMeta(
                name=meta.get("name", filepath.stem),
                version=meta.get("version", "1.0.0"),
                description=meta.get("description", ""),
                author=meta.get("author", ""),
                type=SkillType(meta.get("type", "workspace")),
                triggers=triggers,
                dependencies=meta.get("dependencies", []),
                gepa_generation=meta.get("gepa_generation", 0),
                fitness_score=meta.get("fitness", 0.0),
            )

            return Skill(
                meta=skill_meta,
                content=body.strip(),
                updated_at=filepath.stat().st_mtime,
            )

        except Exception:
            return None

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        """解析 YAML frontmatter。"""
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return {}

        end_idx = -1
        for i in range(1, min(len(lines), 50)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx == -1:
            return {}

        yaml_text = "\n".join(lines[1:end_idx])
        try:
            return yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError:
            return {}

    @staticmethod
    def _extract_body(content: str) -> str:
        """移除 frontmatter 后的正文。"""
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return content

        for i in range(1, min(len(lines), 20)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1:])

        return content
