"""Layer 2: 程序性技能记忆。

任务完成 → 自动分析执行追踪 → 抽象为可复用模式 → 保存技能文件。
遵循 agentskills.io 开放标准。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from soul.types import Skill, SkillMeta, SkillType


class ProceduralMemory:
    """程序性技能记忆 — 第二层。

    自动从成功任务中提取可复用模式，生成 SKILL.md 文件。
    采用语义匹配召回，相似任务自动加载相关技能。
    """

    def __init__(self, skills_dir: str = "~/.soul/skills"):
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self._index: dict[str, list[str]] = {}  # trigger -> skill names

    async def load_all(self) -> list[Skill]:
        """加载所有技能。"""
        self._skills.clear()
        self._index.clear()

        for skill_file in self.skills_dir.glob("*.skill"):
            skill = self._parse_skill_file(skill_file)
            if skill:
                self._skills[skill.meta.name] = skill
                for trigger in skill.meta.triggers:
                    self._index.setdefault(trigger.lower(), []).append(skill.meta.name)

        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def match(self, task_description: str, top_k: int = 3) -> list[Skill]:
        """语义匹配 — 找到与任务最相关的技能。

        当前使用关键词匹配（生产环境可用 embedding 向量检索）。
        """
        task_lower = task_description.lower()
        scored: list[tuple[float, Skill]] = []

        for skill in self._skills.values():
            score = 0.0
            # 触发词匹配：短词直接子串匹配
            for trigger in skill.meta.triggers:
                if trigger.lower() in task_lower:
                    score += 2.0
            # 描述匹配
            desc_words = set(skill.meta.description.lower().split())
            task_words = set(task_lower.split())
            overlap = desc_words & task_words
            score += len(overlap) * 0.5
            # 使用频率加权
            score += min(skill.meta.usage_count / 100, 1.0)
            # 成功率加权（新技能至少给最低权重，避免永不匹配）
            sr = max(skill.meta.success_rate, 0.1)
            score *= sr

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    async def create_from_trace(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
        success: bool = True,
        tool_results: list[Any] | None = None,
    ) -> Skill | None:
        """从执行追踪自动创建技能。

        Args:
            task_description: 任务描述
            execution_trace: 执行步骤记录
            success: 任务是否完全成功
            tool_results: 工具执行结果(用于计算部分成功率)

        Returns:
            新创建的 Skill，如果失败返回 None
        """
        # 至少2个步骤才能学习；即使失败也学（标记为低成功率）
        if len(execution_trace) < 2:
            return None

        # 提取关键步骤
        key_steps = self._extract_key_steps(execution_trace)
        if not key_steps:
            return None

        # 生成技能名称
        name = self._generate_skill_name(task_description)
        if name in self._skills:
            # 更新已有技能
            return await self._update_existing(name, key_steps, task_description)

        # 生成 SKILL.md 内容
        content = self._generate_skill_content(name, task_description, key_steps)
        triggers = self._extract_triggers(task_description)

        # 计算初始成功率：失败任务也能学，但成功率更低
        if success:
            init_success = 1.0
        elif tool_results:
            ok = sum(1 for r in tool_results if getattr(r, 'success', False))
            total = max(len(tool_results), 1)
            init_success = max(0.1, ok / total)
        else:
            init_success = 0.3  # 失败无数据，给最低信任分

        meta = SkillMeta(
            name=name,
            version="1.0.0",
            description=f"自动生成的技能：{task_description[:100]}",
            type=SkillType.EVOLVED,
            triggers=triggers,
            success_rate=init_success,
        )

        skill = Skill(meta=meta, content=content)

        # 保存到磁盘
        self._save_skill_file(skill)
        self._skills[name] = skill
        for t in triggers:
            self._index.setdefault(t.lower(), []).append(name)

        return skill

    async def _update_existing(
        self, name: str, key_steps: list[str], description: str
    ) -> Skill:
        """更新已有技能。"""
        skill = self._skills[name]
        # 增量改进：追加新发现的关键步骤
        existing_steps = set(skill.content.split("\n"))
        new_steps = [s for s in key_steps if s not in existing_steps]
        if new_steps:
            skill.content += "\n\n## 补充步骤\n" + "\n".join(new_steps)
            # 迭代版本
            parts = skill.meta.version.split(".")
            skill.meta.version = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
            skill.updated_at = time.time()
            self._save_skill_file(skill)
        skill.meta.usage_count += 1
        # 每次成功复用：提升适应度 + 推进 GEPA 代际
        skill.meta.fitness_score = min(1.0, skill.meta.fitness_score + 0.1)
        skill.meta.success_rate = min(1.0, skill.meta.success_rate + 0.05)
        # 每使用3次推进一代，标记可触发 GEPA 进化
        if skill.meta.usage_count % 3 == 0:
            skill.meta.gepa_generation += 1
        return skill

    def _extract_key_steps(self, trace: list[dict[str, Any]]) -> list[str]:
        """从执行追踪中提取关键步骤。"""
        steps: list[str] = []
        for item in trace:
            action = item.get("action", "")
            if action in ("tool_call", "bash", "file_write", "decision"):
                desc = item.get("description", "") or str(item.get("result", ""))[:200]
                steps.append(f"- {action}: {desc}")
        return steps

    def _generate_skill_name(self, description: str) -> str:
        """从描述生成安全的技能名称（最多 50 字符，无路径分隔符）。"""
        import re
        # 移除路径、URL、特殊字符（含中文全角标点），只保留中英文和数字
        clean = re.sub(r'[\\/:*?"<>|：；（）【】！，。？、￥…—]', '', description)
        clean = re.sub(r'\s+', '_', clean.strip())
        # 压缩连续下划线
        clean = re.sub(r'_+', '_', clean)
        # 取前 50 个字符
        clean = clean[:50].strip('_')
        # 确保至少有意义的名称
        if len(clean) < 3:
            import hashlib
            clean = "skill_" + hashlib.md5(description.encode()).hexdigest()[:8]
        return clean

    def _generate_skill_content(
        self, name: str, description: str, steps: list[str]
    ) -> str:
        """生成 SKILL.md 内容。"""
        return f"""# {name}

## 描述
{description}

## 触发条件
{self._extract_triggers(description)}

## 执行步骤
{chr(10).join(steps)}

## 注意事项
- 此技能由 AI 自动生成，使用前请验证步骤适用性
- 首次使用成功率可能较低，会随使用迭代改进
"""

    def _extract_triggers(self, text: str) -> list[str]:
        """从任务描述中提取触发关键词（中英文）。"""
        # 高频技术关键词
        tech_keywords = [
            # 英文
            "docker", "git", "python", "node", "react", "vue", "api",
            "database", "deploy", "test", "build", "install", "config",
            "backup", "html", "css", "javascript", "markdown", "frontend",
            "web", "editor", "preview", "file", "write", "read", "server",
            # 中文
            "部署", "测试", "安装", "配置", "备份", "数据库", "日志",
            "前端", "后端", "编辑器", "预览", "小说", "写作", "章节",
            "导出", "导入", "优化", "重构", "修复", "创建", "文件",
            "代码", "项目", "工具", "管理", "主题", "样式", "暗色",
        ]
        text_lower = text.lower()
        found = []
        for k in tech_keywords:
            if k in text_lower:
                found.append(k)
                if len(found) >= 8:  # 最多 8 个触发词
                    break
        return found

    def _parse_skill_file(self, filepath: Path) -> Skill | None:
        """解析技能文件。"""
        try:
            raw_content = filepath.read_text(encoding="utf-8")
            name = filepath.stem

            # 解析 frontmatter
            body, frontmatter = self._split_frontmatter(raw_content)
            if frontmatter:
                name = frontmatter.get("name", name)
                description = frontmatter.get("description", "")
                version = str(frontmatter.get("version", "1.0.0"))
                triggers = frontmatter.get("triggers", [])
                if isinstance(triggers, str):
                    triggers = [t.strip() for t in triggers.split(",")]
                gepa_gen = int(frontmatter.get("gepa_generation", 0))
                fitness = float(frontmatter.get("fitness", 0.0))
            else:
                description = ""
                version = "1.0.0"
                triggers = []
                gepa_gen = 0
                fitness = 0.0

            meta = SkillMeta(
                name=name,
                version=version,
                description=description,
                type=SkillType.EVOLVED if gepa_gen > 0 else SkillType.WORKSPACE,
                triggers=triggers,
                gepa_generation=gepa_gen,
                fitness_score=fitness,
            )

            # 存储 BODY ONLY（不含 frontmatter），防止重复 frontmatter
            return Skill(meta=meta, content=body.strip())

        except Exception:
            return None

    def _split_frontmatter(self, content: str) -> tuple[str, dict]:
        """分离 frontmatter 和正文。"""
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return content, {}

        end_idx = -1
        for i in range(1, min(len(lines), 50)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx == -1:
            return content, {}

        import yaml
        yaml_text = "\n".join(lines[1:end_idx])
        try:
            frontmatter = yaml.safe_load(yaml_text) or {}
        except Exception:
            frontmatter = {}

        body = "\n".join(lines[end_idx + 1:])
        return body, frontmatter

    def _save_skill_file(self, skill: Skill) -> None:
        """保存技能文件到磁盘。content 不含 frontmatter，此处添加。"""
        filepath = self.skills_dir / f"{skill.meta.name}.skill"
        header = f"""---
name: {skill.meta.name}
version: {skill.meta.version}
description: {skill.meta.description}
triggers: {', '.join(skill.meta.triggers)}
gepa_generation: {skill.meta.gepa_generation}
fitness: {skill.meta.fitness_score}
---

"""
        filepath.write_text(header + skill.content, encoding="utf-8")

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.meta.name,
                "version": s.meta.version,
                "description": s.meta.description,
                "triggers": s.meta.triggers,
                "fitness": s.meta.fitness_score,
                "usage": s.meta.usage_count,
            }
            for s in self._skills.values()
        ]
