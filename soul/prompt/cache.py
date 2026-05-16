"""Prompt 前缀缓存管理。

冻结快照机制：会话开始时冻结 system prompt 内容，
会话中修改只写磁盘不更新 prompt，保护 LLM prefix cache。
下次会话自动加载新版本。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any


class PrefixCache:
    """管理 LLM prefix cache 的冻结快照机制。

    核心原理：
    - 会话开始时读取配置文件，计算哈希
    - 将内容冻结注入 system prompt
    - 会话中修改只写入磁盘，不影响当前 prompt
    - 下次会话自动使用新版本（cache 自然失效）
    """

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()
        self._snapshots: dict[str, tuple[str, str]] = {}  # file_path -> (content, hash)
        self._frozen = False

    def freeze(self) -> None:
        """冻结当前快照 — 会话中不再更新。"""
        self._frozen = True

    def thaw(self) -> None:
        """解冻 — 下次读取时重新加载。"""
        self._frozen = False
        self._snapshots.clear()

    def read(self, filename: str) -> str:
        """读取文件内容（优先使用快照）。"""
        filepath = self.workspace / filename
        cache_key = str(filepath)

        if cache_key in self._snapshots:
            return self._snapshots[cache_key][0]

        content = self._read_file(filepath)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        self._snapshots[cache_key] = (content, content_hash)
        return content

    def write(self, filename: str, content: str) -> None:
        """写入文件并更新快照（如果未冻结）。"""
        filepath = self.workspace / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

        cache_key = str(filepath)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if not self._frozen:
            self._snapshots[cache_key] = (content, content_hash)

    def get_hash(self, filename: str) -> str:
        """获取文件内容的哈希。"""
        filepath = self.workspace / filename
        cache_key = str(filepath)
        if cache_key in self._snapshots:
            return self._snapshots[cache_key][1]
        return ""

    def invalidate(self, filename: str | None = None) -> None:
        """使缓存失效。"""
        if filename:
            cache_key = str(self.workspace / filename)
            self._snapshots.pop(cache_key, None)
        else:
            self._snapshots.clear()

    @staticmethod
    def _read_file(filepath: Path) -> str:
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return ""

    def get_snapshot_info(self) -> dict[str, Any]:
        """获取快照信息（用于调试）。"""
        return {
            "frozen": self._frozen,
            "files": {
                Path(k).name: {"hash": v[1], "size": len(v[0])}
                for k, v in self._snapshots.items()
            },
        }
