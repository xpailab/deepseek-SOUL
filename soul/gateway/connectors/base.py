"""平台连接器抽象基类。

定义统一的平台连接器接口，所有平台（QQ/微信/钉钉/飞书/Telegram等）
继承此类实现具体平台的连接、消息收发和监听逻辑。
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = ["PlatformConnector"]


class PlatformConnector:
    """平台连接器基类 — 统一消息收发接口。

    子类实现具体平台:
        QQConnector, WeChatConnector, DingTalkConnector,
        FeishuConnector, TelegramConnector
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self._connected: bool = False
        self._handlers: dict[str, list] = {"message": [], "error": [], "connect": []}

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """建立连接。子类实现。"""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """断开连接。子类实现。"""
        raise NotImplementedError

    async def send_message(self, content: str, chat_id: str, **kwargs) -> str:
        """发送消息。子类实现。"""
        raise NotImplementedError

    async def listen(self) -> None:
        """开始监听消息。子类实现。"""
        raise NotImplementedError

    def on(self, event: str, handler) -> None:
        """注册事件处理器。"""
        self._handlers.setdefault(event, []).append(handler)

    async def _emit(self, event: str, **data) -> None:
        """触发事件。"""
        for handler in self._handlers.get(event, []):
            try:
                result = handler(**data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
