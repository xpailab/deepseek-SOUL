"""4 层记忆系统 — 超越 Hermes 的 3 层设计。

Layer 1: 冻结快照 (Frozen) — MEMORY.md + USER.md，保护 prefix cache
Layer 2: 程序技能 (Procedural) — 自动生成 SKILL.md，可复用模式
Layer 3: 混合检索 (Indexed) — FTS5 全文搜索 + LLM 语义理解
Layer 4: 预测记忆 (Predictive) — 预测用户意图，主动准备上下文 [SOUL 创新]
"""
from soul.memory.frozen import FrozenMemory
from soul.memory.indexed import IndexedMemory
from soul.memory.manager import MemoryManager
from soul.memory.predictive import PredictiveMemory
from soul.memory.procedural import ProceduralMemory

__all__ = [
    "FrozenMemory",
    "ProceduralMemory",
    "IndexedMemory",
    "PredictiveMemory",
    "MemoryManager",
]
