"""Prompt 系统测试。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from soul.types import Skill, SkillMeta, SkillType


class TestPromptBuilder:
    def test_build_empty_prompt(self):
        from soul.prompt.builder import PromptBuilder
        tmp = tempfile.mkdtemp(prefix="soul_prompt_")
        workspace = Path(tmp) / "workspace"
        workspace.mkdir(parents=True)

        builder = PromptBuilder(
            workspace_dir=str(workspace),
            skills_dir=str(workspace / "skills"),
            soul_file="",
            identity_file="",
        )
        prompt = builder.build_system_prompt(matched_skills=[], tools=[], extra_context="")
        assert isinstance(prompt, str)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_build_with_tools(self):
        from soul.prompt.builder import PromptBuilder
        tmp = tempfile.mkdtemp(prefix="soul_prompt_")
        workspace = Path(tmp) / "workspace"
        workspace.mkdir(parents=True)

        builder = PromptBuilder(
            workspace_dir=str(workspace),
            skills_dir=str(workspace / "skills"),
            soul_file="",
            identity_file="",
        )
        tools = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object", "properties": {}}},
        ]
        prompt = builder.build_system_prompt(matched_skills=[], tools=tools, extra_context="")
        assert "test_tool" in prompt
        shutil.rmtree(tmp, ignore_errors=True)

    def test_build_with_skills(self):
        from soul.prompt.builder import PromptBuilder
        tmp = tempfile.mkdtemp(prefix="soul_prompt_")
        workspace = Path(tmp) / "workspace"
        workspace.mkdir(parents=True)

        builder = PromptBuilder(
            workspace_dir=str(workspace),
            skills_dir=str(workspace / "skills"),
            soul_file="",
            identity_file="",
        )
        skill = Skill(
            meta=SkillMeta(name="test_skill", version="1.0.0", description="Test skill"),
            content="## 测试技能\n步骤1: 运行测试",
        )
        prompt = builder.build_system_prompt(matched_skills=[skill], tools=[], extra_context="")
        assert "test_skill" in prompt or "测试技能" in prompt
        shutil.rmtree(tmp, ignore_errors=True)


class TestContextCompressor:
    def test_compressor_init(self):
        from soul.prompt.compressor import ContextCompressor
        c = ContextCompressor()
        assert c.max_tokens > 0
        assert c.safety_margin > 0
