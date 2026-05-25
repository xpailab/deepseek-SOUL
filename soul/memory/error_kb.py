"""错误模式知识库 — 跨会话累积修复经验。

Agent 每次遇到错误并成功修复后，将"错误特征 → 修复方案"存入知识库。
下次遇到相同特征的错误时，直接查找已知修复方案，避免从零排查。

存储: ~/.soul/error_knowledge.json
结构: { "entries": [...], "stats": {...} }
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ErrorEntry:
    """单条错误知识。"""
    signature: str          # 错误特征哈希
    pattern: str            # 匹配模式（正则）
    tool: str               # 产生错误的工具名
    root_cause: str         # 根因
    fix: str                # 修复方案
    usage_count: int = 0    # 使用次数
    success_count: int = 0  # 修复成功次数
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "pattern": self.pattern,
            "tool": self.tool,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ErrorEntry:
        return cls(
            signature=d["signature"],
            pattern=d.get("pattern", ""),
            tool=d.get("tool", ""),
            root_cause=d.get("root_cause", ""),
            fix=d.get("fix", ""),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            created_at=d.get("created_at", time.time()),
            last_used=d.get("last_used", 0),
        )

    @property
    def confidence(self) -> float:
        """置信度 = 成功次数 / 使用次数（新条目默认 0.5）。"""
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count

    @property
    def is_stale(self, days: int = 30) -> bool:
        """超过 days 天未使用视为过期。"""
        if self.last_used == 0:
            return (time.time() - self.created_at) > days * 86400
        return (time.time() - self.last_used) > days * 86400


class ErrorKnowledgeBase:
    """跨会话错误知识库。

    使用方式:
        kb = ErrorKnowledgeBase()
        entry = kb.lookup("docker: Cannot connect to the Docker daemon")
        if entry:
            print(f"已知修复: {entry.fix}")
    """

    MAX_ENTRIES = 200

    def __init__(self, storage_path: str = "~/.soul/error_knowledge.json"):
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.entries: dict[str, ErrorEntry] = {}
        self.stats: dict[str, int] = {
            "total_lookups": 0,
            "total_hits": 0,
            "total_fixes_applied": 0,
            "total_successes": 0,
        }
        self._loaded = False
        self._dirty = False

    # ═══ 持久化 ═══

    def load(self) -> None:
        """从磁盘加载知识库。"""
        if self._loaded:
            return
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                for item in data.get("entries", []):
                    entry = ErrorEntry.from_dict(item)
                    self.entries[entry.signature] = entry
                self.stats = data.get("stats", self.stats)
            except (json.JSONDecodeError, KeyError):
                pass
        self._loaded = True

    def save(self, force: bool = False) -> None:
        """保存知识库到磁盘。无变更时跳过。"""
        if not self._dirty and not force:
            return
        self._dirty = False
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self.entries.values()],
            "stats": self.stats,
            "updated_at": time.time(),
        }
        self.storage_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ═══ 查询 ═══

    def lookup(self, error_text: str, tool: str = "") -> ErrorEntry | None:
        """查找匹配错误文本的已知修复方案。

        匹配策略:
        1. 精确签名匹配（O(1) 哈希查找）
        2. 正则模式匹配（遍历已有模式）
        3. 关键词模糊匹配（工具名 + 错误关键词）

        Returns:
            ErrorEntry if found, None otherwise
        """
        self.stats["total_lookups"] += 1
        if not self.entries:
            return None

        error_lower = error_text.lower()

        # 1. 精确签名
        sig = self._make_signature(error_text)
        if sig in self.entries:
            entry = self.entries[sig]
            entry.usage_count += 1
            entry.last_used = time.time()
            self.stats["total_hits"] += 1
            return entry

        # 2. 正则模式匹配
        for entry in self.entries.values():
            if entry.pattern and re.search(entry.pattern, error_lower):
                if tool and entry.tool and entry.tool != tool:
                    continue  # 工具不匹配则跳过
                entry.usage_count += 1
                entry.last_used = time.time()
                self.stats["total_hits"] += 1
                return entry

        # 3. 关键词模糊匹配
        keywords = self._extract_keywords(error_text)
        for entry in self.entries.values():
            entry_kw = self._extract_keywords(entry.pattern)
            common = keywords & entry_kw
            if len(common) >= 2:
                if tool and entry.tool and entry.tool != tool:
                    continue
                entry.usage_count += 1
                entry.last_used = time.time()
                self.stats["total_hits"] += 1
                return entry

        return None

    def lookup_by_confidence(self, error_text: str, min_confidence: float = 0.6) -> ErrorEntry | None:
        """仅返回高置信度的修复方案。"""
        entry = self.lookup(error_text)
        if entry and entry.confidence >= min_confidence:
            return entry
        return None

    # ═══ 学习 ═══

    def learn(
        self,
        error_text: str,
        tool: str,
        root_cause: str = "",
        fix: str = "",
        pattern: str = "",
    ) -> ErrorEntry:
        """从一次错误和修复中学习。

        Args:
            error_text: 错误信息原文
            tool: 产生错误的工具名
            root_cause: 根因分析
            fix: 有效的修复方案
            pattern: 可选的匹配模式（正则）

        Returns:
            新建或更新的 ErrorEntry
        """
        sig = self._make_signature(error_text)
        if sig in self.entries:
            entry = self.entries[sig]
            entry.usage_count += 1
            entry.last_used = time.time()
            if fix and fix != entry.fix:
                entry.fix = fix  # 更新为更有效的修复
            if root_cause:
                entry.root_cause = root_cause
        else:
            entry = ErrorEntry(
                signature=sig,
                pattern=pattern or self._derive_pattern(error_text),
                tool=tool,
                root_cause=root_cause,
                fix=fix,
            )
            self.entries[sig] = entry

        self._dirty = True
        # 裁剪
        self._prune()
        return entry

    def record_result(self, error_text: str, success: bool) -> None:
        """记录修复结果——更新置信度。"""
        sig = self._make_signature(error_text)
        if sig in self.entries:
            self.entries[sig].success_count += 1 if success else 0
            self.stats["total_fixes_applied"] += 1
            self._dirty = True
            if success:
                self.stats["total_successes"] += 1

    # ═══ 维护 ═══

    def _prune(self) -> None:
        """裁剪过大的知识库——移除过期和低质量条目。"""
        if len(self.entries) <= self.MAX_ENTRIES:
            return

        # 先移除过期条目
        stale = [k for k, e in self.entries.items() if e.is_stale]
        for k in stale:
            del self.entries[k]

        # 如果还是太多，移除低置信度条目
        if len(self.entries) > self.MAX_ENTRIES:
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: (x[1].confidence, x[1].usage_count),
            )
            to_remove = sorted_entries[:len(self.entries) - self.MAX_ENTRIES]
            for k, _ in to_remove:
                del self.entries[k]

    # ═══ 工具方法 ═══

    @staticmethod
    def _make_signature(error_text: str) -> str:
        """为错误文本生成特征签名——标准化后哈希。"""
        # 移除变量部分：路径、IP、端口、时间戳、数字
        normalized = re.sub(r'/[^\s]*/[^\s]*', '/<path>', error_text)
        normalized = re.sub(r'\d+\.\d+\.\d+\.\d+(:\d+)?', '<ip>', normalized)
        normalized = re.sub(r'0x[0-9a-fA-F]+', '<hex>', normalized)
        normalized = re.sub(r'\b\d{10,}\b', '<ts>', normalized)
        normalized = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<date>', normalized)
        normalized = normalized.lower().strip()[:200]
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _derive_pattern(error_text: str) -> str:
        """从错误文本导出通用正则模式。"""
        text = error_text.lower()[:300]
        key = re.escape(text.split('\n')[0][:80])
        # 宽松化：允许任意中间字符
        parts = key.split(r'\ ')
        if len(parts) >= 3:
            return r'\b' + r'\s+'.join(parts[:3])
        return key

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """提取关键词（最小长度 3）。"""
        words = re.findall(r'[a-z0-9_]+', text.lower())
        return {w for w in words if len(w) >= 3 and w not in ("the", "and", "for", "was", "has", "that", "this", "with")}

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self.entries),
            "high_confidence": sum(1 for e in self.entries.values() if e.confidence >= 0.7),
            **self.stats,
            "hit_rate": round(self.stats["total_hits"] / max(1, self.stats["total_lookups"]) * 100, 1),
        }
