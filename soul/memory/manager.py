"""记忆管理器 — 统一管理 4 层记忆系统。

协调各层记忆的读写、检索、压缩和生命周期。
提供统一 API，内部智能路由到最合适的记忆层。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from soul.memory.frozen import FrozenMemory
from soul.memory.indexed import IndexedMemory
from soul.memory.predictive import PredictiveMemory
from soul.memory.procedural import ProceduralMemory
from soul.memory.user_model import UserModel
from soul.types import (
    MemoryConfig,
    MemoryEntry,
    MemoryLayer,
    Message,
    Skill,
)


class MemoryManager:
    """统一记忆管理器。

    四层架构：
    Layer 1 (Frozen):  系统提示记忆 — 即时注入，prefix cache 保护
    Layer 2 (Procedural): 技能记忆 — 自动生成，语义召回
    Layer 3 (Indexed):  对话记忆 — FTS5 + LLM 混合检索
    Layer 4 (Predictive): 预测记忆 — 行为预测，主动建议 [SOUL 创新]
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        skills_dir = str(Path(self.config.workspace_dir).expanduser().parent / "skills")
        self.frozen = FrozenMemory(str(self.config.workspace_dir))
        self.procedural = ProceduralMemory(skills_dir)
        self.indexed = IndexedMemory(str(self.config.fts_db_path))
        self.predictive = PredictiveMemory()
        self.user_model = UserModel(str(self.config.workspace_dir))
        self.peers = self.user_model.peers  # 快捷访问
        self._initialized = False

    def set_llm(self, llm: Callable[[str], Awaitable[str]]) -> None:
        """注入 LLM 回调，启用 Layer 3 的动态查询扩展和语义重排。

        llm 签名: async def llm(prompt: str) -> str

        示例:
            async def my_llm(prompt: str) -> str:
                return await openai_adapter.chat(prompt)
            memory_manager.set_llm(my_llm)
        """
        self._llm = llm
        self.indexed.set_llm(llm)

    async def initialize(self) -> None:
        """初始化所有记忆层。"""
        if self._initialized:
            return
        # L2: 先加载用户技能（~/.soul/skills/），再叠加载入内置技能
        await self.procedural.load_all()
        await self._load_bundled_skills()
        self.frozen.snapshot()
        self.predictive.load()
        self._initialized = True

    async def _load_bundled_skills(self) -> None:
        """加载项目内置捆绑技能到 ProceduralMemory。"""
        bundled = Path(__file__).resolve().parent.parent.parent / "skills" / "bundled"
        if not bundled.exists():
            return
        from soul.skills.loader import SkillLoader
        loader = SkillLoader(str(self.procedural.skills_dir), str(bundled))
        skills = await loader.load_all()
        for s in skills:
            if not self.procedural.get(s.meta.name):
                self.procedural._skills[s.meta.name] = s
                for t in s.meta.triggers:
                    self.procedural._index.setdefault(t.lower(), []).append(s.meta.name)

    # ═══════════════════════════════════════════
    # 统一查询 API
    # ═══════════════════════════════════════════

    async def query(
        self,
        query: str,
        layers: list[MemoryLayer] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """统一查询 — 智能路由到最合适的记忆层。

        Returns:
            {
                "frozen": {...},       # Layer 1 结果
                "procedural": [...],   # Layer 2 匹配的技能
                "indexed": [...],      # Layer 3 FTS5 + 语义结果
                "predictive": [...],   # Layer 4 预测
            }
        """
        if not layers:
            layers = list(MemoryLayer)

        result: dict[str, Any] = {}

        if MemoryLayer.FROZEN in layers:
            result["frozen"] = {
                "memory": self.frozen.get_memory(),
                "user": self.frozen.get_user(),
                "usage": self.frozen.get_usage(),
            }

        if MemoryLayer.PROCEDURAL in layers:
            result["procedural"] = self.procedural.match(query, top_k=limit)

        if MemoryLayer.INDEXED in layers:
            fts_results = await self.indexed.search_fts(query, limit=limit)
            structured = await self.indexed.retrieve_memories(query, limit=limit)
            result["indexed"] = {
                "conversations": fts_results,
                "memories": [m.model_dump() for m in structured],
            }

        if MemoryLayer.PREDICTIVE in layers:
            predictions = await self.predictive.predict_next_actions(query)
            result["predictive"] = predictions

        return result

    async def query_for_prompt(
        self,
        query: str = "",
        context: dict[str, Any] | None = None,
    ) -> str:
        """查询并生成可直接注入 prompt 的字符串。

        自动从各层检索最相关的信息，格式化为 prompt 片段。
        """
        sections: list[str] = []

        # Layer 1: 冻结记忆（已在 system prompt 中，这里不重复）

        # Layer 2: 匹配的技能
        if query:
            skills = self.procedural.match(query, top_k=2)
            if skills:
                for s in skills:
                    sections.append(
                        f"[相关技能: {s.meta.name}]\n{s.content[:500]}"
                    )

        # Layer 3: FTS5 检索
        if query:
            # 先用静态同义词表搜索（0 延迟）
            conv_results = await self.indexed.search_semantic(query, limit=3, use_llm=False)
            # 无结果且查询有意义时，用 LLM 扩展再试
            if not conv_results and len(query) > 10:
                conv_results = await self.indexed.search_semantic(query, limit=3, use_llm=True)
            if conv_results:
                lines = ["[历史相关对话]"]
                for r in conv_results:
                    lines.append(f"- [{r['role']}] {r['content'][:200]}")
                sections.append("\n".join(lines))

        # Layer 4: 预测上下文
        if self.config.predictive_enabled:
            pred_prompt = await self.predictive.get_predictive_context_prompt()
            if pred_prompt:
                sections.append(pred_prompt)

        # 用户模型 — Multi-Peer 角色 + 辩证推理画像
        user_fragment = self.user_model.get_full_prompt_fragment()
        if user_fragment:
            sections.append(user_fragment)

        return "\n\n".join(sections)

    # ═══════════════════════════════════════════
    # 记忆操作 API
    # ═══════════════════════════════════════════

    async def remember(
        self,
        content: str,
        layer: MemoryLayer = MemoryLayer.FROZEN,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        """存储记忆到指定层。"""
        if layer == MemoryLayer.FROZEN:
            self.frozen.add("MEMORY.md", content)
            return "frozen"

        elif layer == MemoryLayer.INDEXED:
            entry = MemoryEntry(
                layer=layer,
                content=content,
                importance=importance,
                tags=tags or [],
            )
            await self.indexed.store_memory_entry(entry)
            return entry.id

        return ""

    async def forget(
        self,
        pattern: str,
        layer: MemoryLayer = MemoryLayer.FROZEN,
    ) -> None:
        """遗忘记忆。"""
        if layer == MemoryLayer.FROZEN:
            self.frozen.remove("MEMORY.md", pattern)

    async def store_conversation(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        """批量存储对话消息。"""
        for msg in messages:
            await self.indexed.store_message(
                session_id=session_id,
                role=msg.role.value,
                content=msg.content,
            )

    async def learn_skill(
        self,
        task: str,
        trace: list[dict[str, Any]],
        success: bool = True,
        tool_results: list[Any] | None = None,
    ) -> Skill | None:
        """从任务执行中学习新技能（Layer 2）。"""
        return await self.procedural.create_from_trace(task, trace, success, tool_results)

    async def observe_action(
        self,
        action: str,
        previous: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """观察用户行为（Layer 4 + 用户模型）。"""
        if self.config.predictive_enabled:
            await self.predictive.observe(action, previous, context)
        self.user_model.observe_message(action, role="user")

    # ═══════════════════════════════════════════
    # 维护 API
    # ═══════════════════════════════════════════

    async def compact(self) -> dict[str, Any]:
        """执行全系统记忆压缩。"""
        result: dict[str, Any] = {}

        # Layer 1: 检查容量
        usage = self.frozen.get_usage()
        if usage["memory"]["pct"] > 80:
            current = self.frozen.get_memory()
            compressed = self.frozen.write("MEMORY.md", current)
            result["frozen_compressed"] = True

        # Layer 3: 清理旧对话
        deleted = await self.indexed.compact_old_sessions(30)
        result["deleted_conversations"] = deleted

        return result

    async def get_stats(self) -> dict[str, Any]:
        """获取全部记忆统计信息。"""
        return {
            "frozen": {
                "usage": self.frozen.get_usage(),
                "snapshot_active": self.frozen.is_frozen,
            },
            "procedural": {
                "skill_count": self.procedural.skill_count,
                "skills": self.procedural.list_skills(),
            },
            "indexed": await self.indexed.get_stats(),
            "predictive": {
                "habits": await self.predictive.detect_habits(),
            },
        }

    async def close(self) -> None:
        self.predictive.save()  # 持久化预测数据
        self.user_model.save()  # 持久化用户画像
        await self.indexed.close()
