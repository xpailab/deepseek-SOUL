"""Layer 1: 冻结快照记忆。

会话开始时注入 system prompt，会话中修改只写盘不更新 prompt。
保护 LLM prefix cache，下次会话自动生效。

PrefixCache 已合并到此类 — 统一管理所有 prompt 文件的冻结快照，
避免两套独立实现管理同一批文件的 bug。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class FrozenMemory:
    """冻结快照记忆 — 第一层（含 PrefixCache 功能）。

    管理:
    - MEMORY.md: Agent 笔记本 (≤ 2,200 chars, § 分隔条目)
    - USER.md: 用户画像 (≤ 1,375 chars, § 分隔条目)
    - 任意 prompt 文件 (SOUL.md, IDENTITY.md, AGENTS.md 等)

    冻结机制:
    - 会话开始: snapshot() 读取所有文件 → 计算哈希 → 冻结内容
    - 会话中修改: 立即写盘 → 当前 frozen 内容不变 (prefix cache 保护)
    - 下次会话: 重新 snapshot() 加载新版本
    """

    MAX_MEMORY_CHARS = 2200
    MAX_USER_CHARS = 1375

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._frozen: dict[str, tuple[str, str]] = {}  # filename -> (content, hash)
        self._active = True  # 是否启用冻结

    @property
    def is_frozen(self) -> bool:
        return self._active

    def snapshot(self, extra_files: list[str] | None = None) -> None:
        """拍摄快照 — 冻结当前所有相关文件内容。

        默认冻结 MEMORY.md 和 USER.md（用于记忆管理）。
        可选传入额外文件列表（SOUL.md, IDENTITY.md 等，用于 prompt 构建）。
        """
        files = ["MEMORY.md", "USER.md"] + (extra_files or [])
        for fname in files:
            filepath = self.workspace / fname
            if not filepath.exists():
                filepath.write_text("", encoding="utf-8")
            content = filepath.read_text(encoding="utf-8")
            h = hashlib.sha256(content.encode()).hexdigest()[:16]
            self._frozen[fname] = (content, h)

    def read(self, filename: str) -> str:
        """读取文件（优先返回冻结快照，保护 prefix cache）。"""
        if self._active and filename in self._frozen:
            return self._frozen[filename][0]

        filepath = self.workspace / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            # 不在冻结集合中：缓存但不影响已有的 protection
            if self._active and filename not in self._frozen:
                h = hashlib.sha256(content.encode()).hexdigest()[:16]
                self._frozen[filename] = (content, h)
            return content
        return ""

    def write(self, filename: str, content: str) -> str:
        """写入文件。冻结模式下落盘但不更新快照。"""
        filepath = self.workspace / filename
        max_chars = self.MAX_MEMORY_CHARS if filename == "MEMORY.md" else self.MAX_USER_CHARS

        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        if len(content) > max_chars:
            content = self._compress(content, max_chars)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

        # 非冻结模式或新文件：更新快照
        if not self._active or filename not in self._frozen:
            self._frozen[filename] = (content, h)

        return content

    def add(self, filename: str, entry: str) -> tuple[str, bool]:
        """追加条目到文件（用 § 分隔）。返回 (新内容, 是否压缩)。"""
        current = self.read(filename)
        if current and not current.endswith("\n"):
            current += "\n"
        new_content = current + "§ " + entry.strip()
        max_chars = self.MAX_MEMORY_CHARS if filename == "MEMORY.md" else self.MAX_USER_CHARS
        return self.write(filename, new_content), len(new_content) > max_chars

    def remove(self, filename: str, pattern: str) -> str:
        """移除匹配条目。"""
        current = self.read(filename)
        entries = [e.strip() for e in current.split("§") if e.strip()]
        filtered = [e for e in entries if pattern.strip() not in e]
        new_content = "§ ".join(filtered)
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

    def invalidate(self, filename: str | None = None) -> None:
        """使缓存失效 — 下次 read() 重新从磁盘加载。"""
        if filename:
            self._frozen.pop(filename, None)
        else:
            self._frozen.clear()

    def get_snapshot_info(self) -> dict[str, Any]:
        """获取快照信息（调试用）。"""
        return {
            "frozen": self._active,
            "files": {
                k: {"hash": v[1], "size": len(v[0])}
                for k, v in self._frozen.items()
            },
        }

    @staticmethod
    def _compress(content: str, max_chars: int) -> str:
        """压缩内容到容量限制内，优先保留未完成任务和最新条目。"""
        entries = content.split("§")
        if len(entries) <= 1:
            return content[-max_chars:] if len(content) > max_chars else content

        # 分类：未完成(○) vs 已完成(✓/✗)
        pending = []
        done = []
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            # 检查是否未完成任务标记
            if entry.startswith("[") and "○" in entry[:20]:
                pending.append(entry)
            else:
                done.append(entry)

        # 先保留未完成任务（最重要的）
        total = 0
        kept = []
        for entry in pending:
            size = len(entry) + 3
            if total + size <= max_chars:
                kept.append(entry)
                total += size

        # 再按最新优先保留已完成任务
        for entry in reversed(done):
            size = len(entry) + 3
            if total + size <= max_chars:
                kept.append(entry)
                total += size
            else:
                break

        # kept 里 pending 在前，done 在后（反序），再反转使整体按时间正序
        # pending 保持原序（旧→新），done 是反序加入的（新→旧）
        kept_done = [e for e in kept if e not in pending]
        kept_pending = [e for e in kept if e in pending]
        return "§".join(kept_pending + list(reversed(kept_done)))
