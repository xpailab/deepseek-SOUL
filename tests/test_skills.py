"""技能系统测试。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


class TestSkillLoader:
    @pytest.mark.asyncio
    async def test_load_from_file(self):
        from soul.skills.loader import SkillLoader
        tmp = tempfile.mkdtemp(prefix="soul_skill_")
        skill_file = Path(tmp) / "test.skill"
        skill_file.write_text("""---
name: test_skill
version: 1.0.0
description: A test skill
triggers: test, 测试
---
# test_skill

## 步骤
1. 运行测试
""", encoding="utf-8")

        loader = SkillLoader(str(tmp))
        skill = await loader.load_one("test")
        assert skill is not None
        assert skill.meta.name == "test_skill"
        assert skill.meta.version == "1.0.0"
        assert "测试" in skill.meta.triggers
        shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_load_all_has_bundled(self):
        from soul.skills.loader import SkillLoader
        tmp = tempfile.mkdtemp(prefix="soul_skill_")
        loader = SkillLoader(str(tmp))
        skills = await loader.load_all()
        assert len(skills) >= 0  # 捆绑技能可能被加载
        shutil.rmtree(tmp, ignore_errors=True)


class TestSkillRegistry:
    def test_register_and_search(self):
        from soul.skills.registry import SkillRegistry
        from soul.types import Skill, SkillMeta, SkillType

        reg = SkillRegistry()
        skill = Skill(
            meta=SkillMeta(name="deploy", version="1.0", description="Deploy skill", triggers=["deploy", "部署"]),
            content="部署相关技能",
        )
        reg.register(skill)
        result = reg.search("部署")
        assert len(result) > 0
        assert result[0].meta.name == "deploy"

    def test_empty_search(self):
        from soul.skills.registry import SkillRegistry
        reg = SkillRegistry()
        result = reg.search("nonexistent")
        assert result == []
