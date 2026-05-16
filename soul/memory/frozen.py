"""Layer 1: 冻结快照记忆。

会话开始时注入 system prompt，会话中修改只写盘不更新 prompt。
保护 LLM prefix cache，下次会话自动生效。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path


class FrozenMemory:
    """冻结快照记忆 — 第一层。

    管理:
    - MEMORY.md: Agent 笔记本 (≤ 2,200 chars)
    - USER.md: 用户画像 (≤ 1,375 chars)

    冻结机制:
    - 会话开始: 读取文件 → 计算哈希 → 冻结内容
    - 会话中修改: 立即写盘 → 当前 prompt 不变
    - 下次会话: 自动加载新版本
    """

    MAX_MEMORY_CHARS = 2200
    MAX_USER_CHARS = 1375

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._frozen: dict[str, tuple[str, str]] = {}  # file -> (content, hash)
        self._active = True  # 是否启用冻结

    @property
    def is_frozen(self) -> bool:
        return self._active

    def snapshot(self) -> None:
        """拍摄快照 — 冻结当前内容。"""
        for fname in ["MEMORY.md", "USER.md"]:
            filepath = self.workspace / fname
            if filepath.exists():
                content = filepath.read_text(encoding="utf-8")
                h = hashlib.sha256(content.encode()).hexdigest()[:16]
                self._frozen[fname] = (content, h)
            else:
                self._frozen[fname] = ("", "")

    def read(self, filename: str) -> str:
        """读取文件（优先返回冻结快照）。"""
        if self._active and filename in self._frozen:
            return self._frozen[filename][0]

        filepath = self.workspace / filename
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return ""

    def write(self, filename: str, content: str) -> str:
        """写入文件。返回 (写入后的内容, 是否触发压缩)。"""
        filepath = self.workspace / filename
        max_chars = self.MAX_MEMORY_CHARS if filename == "MEMORY.md" else self.MAX_USER_CHARS

        compressed = False
        if len(content) > max_chars:
            content = self._compress(content, max_chars)
            compressed = True

        filepath.write_text(content, encoding="utf-8")

        if not self._active:
            self._frozen[filename] = (
                content,
                hashlib.sha256(content.encode()).hexdigest()[:16],
            )

        return content

    def add(self, filename: str, entry: str) -> tuple[str, bool]:
        """追加条目到文件。返回 (新内容, 是否压缩)。"""
        current = self.read(filename)
        if current and not current.endswith("\n"):
            current += "\n"
        new_content = current + "§ " + entry.strip()
        return self.write(filename, new_content), len(new_content) > (
            self.MAX_MEMORY_CHARS if filename == "MEMORY.md" else self.MAX_USER_CHARS
        )

    def remove(self, filename: str, pattern: str) -> str:
        """移除匹配条目。"""
        current = self.read(filename)
        entries = current.split("§")
        filtered = [e for e in entries if pattern.strip() not in e]
        new_content = "§".join(filtered).strip()
        return self.write(filename, new_content)

    def get_memory(self) -> str:
        return self.read("MEMORY.md")

    def get_user(self) -> str:
        return self.read("USER.md")

    def get_hash(self, filename: str) -> str:
        if self._active and filename in self._frozen:
            return self._frozen[filename][1]
        content = self.read(filename)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_usage(self) -> dict:
        mem = self.read("MEMORY.md")
        user = self.read("USER.md")
        return {
            "memory": {"chars": len(mem), "max": self.MAX_MEMORY_CHARS, "pct": round(len(mem) / self.MAX_MEMORY_CHARS * 100, 1)},
            "user": {"chars": len(user), "max": self.MAX_USER_CHARS, "pct": round(len(user) / self.MAX_USER_CHARS * 100, 1)},
        }

    @staticmethod
    def _compress(content: str, max_chars: int) -> str:
        """压缩内容到容量限制内，保留最重要条目。"""
        entries = content.split("§")
        if len(entries) <= 1:
            return content[:max_chars]

        kept: list[str] = []
        total = 0
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            size = len(entry) + 3
            if total + size <= max_chars:
                kept.append(entry)
                total += size
            else:
                break
        return "§ ".join(kept)
