"""会话管理器 — 会话生命周期管理。

管理 Agent 会话的创建、恢复、重置和持久化。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from soul.types import (
    AgentState,
    Message,
    MessageRole,
    SandboxMode,
    SessionState,
)


class SessionManager:
    """会话管理器。

    负责:
    - 会话的创建和恢复
    - 会话历史的持久化
    - 会话状态的快照与恢复
    - 多会话隔离
    """

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()
        self.sessions_dir = self.workspace / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        session_key: str = "main",
        sandbox_mode: SandboxMode = SandboxMode.LOCAL,
        metadata: dict[str, Any] | None = None,
    ) -> SessionState:
        """创建新会话。"""
        async with self._lock:
            state = SessionState(
                session_key=session_key,
                sandbox_mode=sandbox_mode,
                metadata=metadata or {},
            )
            self._sessions[state.session_id] = state
            return state

    async def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    async def get_or_create(
        self,
        session_key: str = "main",
        session_id: str = "",
    ) -> SessionState:
        """获取或创建会话。"""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        # 查找同 key 的活跃会话
        for state in self._sessions.values():
            if state.session_key == session_key:
                return state

        return await self.create(session_key=session_key)

    async def add_message(self, session_id: str, message: Message) -> None:
        """添加消息到会话历史。"""
        state = self._sessions.get(session_id)
        if state:
            state.messages.append(message)
            state.message_count = len(state.messages)
            state.token_count += max(1, len(message.content) // 3)
            state.last_active = time.time()

    async def add_messages(self, session_id: str, messages: list[Message]) -> None:
        """批量添加消息 — 只在最后一次更新元数据。"""
        state = self._sessions.get(session_id)
        if not state:
            return
        state.messages.extend(messages)
        state.message_count = len(state.messages)
        state.token_count += sum(max(1, len(m.content) // 3) for m in messages)
        state.last_active = time.time()

    async def get_history(
        self, session_id: str, last_n: int = 0
    ) -> list[Message]:
        """获取会话历史。"""
        state = self._sessions.get(session_id)
        if not state:
            return []
        if last_n > 0:
            return state.messages[-last_n:]
        return state.messages

    async def update_state(self, session_id: str, agent_state: AgentState) -> None:
        state = self._sessions.get(session_id)
        if state:
            state.agent_state = agent_state
            state.last_active = time.time()

    async def save(self, session_id: str) -> str:
        """持久化会话到磁盘（原子写入）。"""
        state = self._sessions.get(session_id)
        if not state:
            return ""

        filepath = self.sessions_dir / f"{session_id}.json"
        tmp_path = filepath.with_suffix(".tmp")
        data = {
            "session_id": state.session_id,
            "session_key": state.session_key,
            "agent_state": state.agent_state.value,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp,
                }
                for m in state.messages
            ],
            "created_at": state.created_at,
            "last_active": state.last_active,
            "message_count": state.message_count,
            "token_count": state.token_count,
            "sandbox_mode": state.sandbox_mode.value,
            "metadata": state.metadata,
        }
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(filepath)  # 原子替换
        return str(filepath)

    async def restore(self, session_id: str) -> SessionState | None:
        """从磁盘恢复会话。"""
        filepath = self.sessions_dir / f"{session_id}.json"
        if not filepath.exists():
            return None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        try:
            state = SessionState(
                session_id=data.get("session_id", session_id),
                session_key=data.get("session_key", "main"),
                agent_state=AgentState(data.get("agent_state", "idle")),
                created_at=data.get("created_at", time.time()),
                last_active=data.get("last_active", time.time()),
                message_count=data.get("message_count", 0),
                token_count=data.get("token_count", 0),
                sandbox_mode=SandboxMode(data.get("sandbox_mode", "local")),
                metadata=data.get("metadata", {}),
            )

            for m in data.get("messages", []):
                try:
                    state.messages.append(Message(
                        id=m.get("id", ""),
                        role=MessageRole(m.get("role", "user")),
                        content=m.get("content", ""),
                        timestamp=m.get("timestamp", time.time()),
                    ))
                except (ValueError, KeyError):
                    continue
        except (ValueError, KeyError):
            return None

        self._sessions[state.session_id] = state
        return state

    async def reset(self, session_id: str) -> SessionState | None:
        """重置会话（清空历史，保留 session_id）。"""
        state = self._sessions.get(session_id)
        if state:
            state.messages.clear()
            state.message_count = 0
            state.token_count = 0
            state.agent_state = AgentState.IDLE
            state.last_active = time.time()
        return state

    async def close(self, session_id: str) -> None:
        """关闭会话。"""
        try:
            await self.save(session_id)
        except Exception:
            pass  # 保存失败不阻止会话关闭
        self._sessions.pop(session_id, None)

    async def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话。"""
        result: list[dict[str, Any]] = []
        for state in self._sessions.values():
            result.append({
                "session_id": state.session_id,
                "session_key": state.session_key,
                "state": state.agent_state.value,
                "messages": state.message_count,
                "tokens": state.token_count,
                "created": state.created_at,
                "last_active": state.last_active,
            })
        result.sort(key=lambda x: x["last_active"], reverse=True)
        return result

    async def spawn_subagent(
        self, parent_session_id: str, task: str
    ) -> SessionState:
        """创建子会话用于子 Agent。"""
        return await self.create(
            session_key=f"subagent_{parent_session_id[:8]}",
            metadata={"parent_session": parent_session_id, "task": task},
        )

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def close_all(self) -> None:
        for sid in list(self._sessions.keys()):
            try:
                await self.close(sid)
            except Exception:
                pass  # 单个会话关闭失败不阻止其他会话
