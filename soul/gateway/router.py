"""消息路由器 — 统一消息格式转换与路由。

所有外部消息先转成统一内部格式，再路由到 Agent 处理器。
"""

from __future__ import annotations

import time
from typing import Any

from soul.types import Message, MessageRole, QueueMode


class ChannelMessage:
    """通道消息 — 统一内部格式。"""

    def __init__(
        self,
        raw_text: str,
        channel: str,
        channel_user_id: str,
        channel_user_name: str = "",
        session_id: str = "",
        is_dm: bool = True,
        attachments: list[dict[str, Any]] | None = None,
        reply_to: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.id = f"chmsg_{int(time.time() * 1000)}"
        self.raw_text = raw_text
        self.channel = channel  # telegram, discord, slack, wechat, cli...
        self.channel_user_id = channel_user_id
        self.channel_user_name = channel_user_name
        self.session_id = session_id or f"{channel}:{channel_user_id}"
        self.is_dm = is_dm
        self.attachments = attachments or []
        self.reply_to = reply_to
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_message(self) -> Message:
        """转为内部 Message 格式。"""
        return Message(
            role=MessageRole.USER,
            content=self.raw_text,
            metadata={
                "channel": self.channel,
                "user_id": self.channel_user_id,
                "user_name": self.channel_user_name,
                "is_dm": self.is_dm,
                "source_msg_id": self.id,
            },
            timestamp=self.timestamp,
        )

    def resolve_queue_mode(self, text: str) -> QueueMode:
        """根据消息内容自动判断队列模式。"""
        text_lower = text.lower().strip()

        # 紧急停止
        if text_lower in ("stop", "停止", "取消", "abort", "cancel"):
            return QueueMode.INTERRUPT

        # 即时指令（短消息，通常是对当前输出的纠正）
        if len(text) < 50 and text_lower.startswith(("用", "use ", "不要", "don't", "等等", "wait")):
            return QueueMode.STEER

        # 默认
        return QueueMode.ADAPTIVE


class MessageRouter:
    """消息路由器。"""

    def __init__(self):
        self._handlers: dict[str, Any] = {}

    def register_handler(self, channel: str, handler: Any) -> None:
        """注册消息处理器。"""
        self._handlers[channel] = handler

    async def route(self, msg: ChannelMessage) -> dict[str, Any]:
        """路由消息到对应处理器。"""
        handler = self._handlers.get(msg.channel)
        if handler:
            return await handler(msg)
        return {"error": f"未找到通道处理器: {msg.channel}"}

    def get_session_id(self, msg: ChannelMessage) -> str:
        """获取会话 ID。"""
        return msg.session_id
