"""跨进程会话同步 — 使 CLI 和 Web UI 共享会话状态。

机制: 基于文件轮询的会话变更检测。
- 每个会话保存为 JSON 文件到 sessions_dir/
- 进程通过检查文件 mtime 检测其他进程的变更
- 轻量级，无需 Redis/消息队列
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any


class SessionSync:
    """跨进程会话同步。

    使用:
        sync = SessionSync("~/.soul/workspace/sessions")
        await sync.start()
        # ... 其他进程修改了会话 ...
        changed = await sync.poll_changes()  # 检测变更
    """

    def __init__(self, sessions_dir: str = "~/.soul/workspace/sessions"):
        self.sessions_dir = Path(sessions_dir).expanduser().resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._known_mtimes: dict[str, float] = {}  # session_id → mtime
        self._running: bool = False
        self._poll_interval: float = 2.0  # 轮询间隔（秒）
        self._listeners: list = []

    async def start(self) -> None:
        """启动同步服务。"""
        self._running = True
        self._scan_sessions()  # 初始扫描

    async def stop(self) -> None:
        self._running = False

    def _scan_sessions(self) -> dict[str, float]:
        """扫描所有会话文件的修改时间。"""
        mtimes: dict[str, float] = {}
        for f in self.sessions_dir.glob("*.json"):
            sid = f.stem
            mtimes[sid] = f.stat().st_mtime
        return mtimes

    async def poll_changes(self) -> dict[str, str]:
        """轮询检测变更的会话。

        Returns:
            {session_id: change_type}  — "created", "modified", "deleted"
        """
        current = self._scan_sessions()
        changes: dict[str, str] = {}

        # 新增的会话
        for sid in current:
            if sid not in self._known_mtimes:
                changes[sid] = "created"

        # 修改的会话
        for sid in current:
            if sid in self._known_mtimes:
                if current[sid] > self._known_mtimes[sid]:
                    changes[sid] = "modified"

        # 删除的会话（在已知但不在当前）
        for sid in self._known_mtimes:
            if sid not in current:
                changes[sid] = "deleted"

        self._known_mtimes = current
        return changes

    def notify_change(self, session_id: str) -> None:
        """通知其他进程：本进程修改了会话。"""
        # 简单实现：touch 会话文件更新时间戳
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.touch()
        self._known_mtimes[session_id] = session_file.stat().st_mtime \
            if session_file.exists() else time.time()

    async def sync_loop(self, callback) -> None:
        """后台轮询循环 — 检测到变更时调用 callback(session_id, change_type)。"""
        self._scan_sessions()
        while self._running:
            changes = await self.poll_changes()
            for sid, change_type in changes.items():
                try:
                    result = callback(sid, change_type)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
            await asyncio.sleep(self._poll_interval)

    def get_shared_session(self, session_id: str) -> dict[str, Any] | None:
        """读取共享的会话数据。"""
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        try:
            return json.loads(session_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def write_shared_session(self, session_id: str, data: dict[str, Any]) -> None:
        """写入共享的会话数据。"""
        session_file = self.sessions_dir / f"{session_id}.json"
        session_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list_shared_sessions(self) -> list[str]:
        """列出所有共享会话 ID。"""
        return [f.stem for f in self.sessions_dir.glob("*.json")]


# ═══════════════════════════════════════════
# 平台连接器 — 重新导出（向后兼容）
# ═══════════════════════════════════════════

from soul.gateway.connectors.base import PlatformConnector  # noqa: F401
