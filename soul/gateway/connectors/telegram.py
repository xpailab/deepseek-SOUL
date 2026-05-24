"""Telegram 连接器 — Bot API。

接入方式:
1. @BotFather 创建机器人 → 获取 API Token
2. 支持长轮询 (getUpdates) + Webhook 两种模式
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from soul.gateway.connectors.base import PlatformConnector


class TelegramConnector(PlatformConnector):
    """Telegram Bot 连接器。

    使用:
        tg = TelegramConnector(bot_token="123456:ABC-DEF")
        await tg.connect()
        await tg.send_message("你好", chat_id="123456789")

        # 长轮询模式
        tg.on("message", lambda chat_id, text: print(f"{chat_id}: {text}"))
        await tg.listen()
    """

    API_BASE = "https://api.telegram.org/bot"

    def __init__(self, bot_token: str = "", config: dict[str, Any] | None = None):
        super().__init__("telegram", config)
        self.bot_token = bot_token
        self._http: httpx.AsyncClient | None = None
        self._last_update_id: int = 0
        self._webhook_url: str = ""

    async def connect(self) -> bool:
        try:
            self._http = httpx.AsyncClient(timeout=30.0)
            # 验证 Token
            resp = await self._http.get(f"{self.API_BASE}{self.bot_token}/getMe")
            data = resp.json()
            if data.get("ok"):
                self._connected = True
                await self._emit("connect", platform="telegram", bot_name=data["result"]["username"])
                return True
            return False
        except Exception as e:
            await self._emit("error", platform="telegram", error=str(e))
            return False

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False

    async def send_message(
        self, content: str, chat_id: str, msg_type: str = "text", **kwargs
    ) -> str:
        """发送消息到 Telegram。

        Args:
            content: 消息内容
            chat_id: 用户/群组/频道 ID
            msg_type: text / markdown / html
            reply_to: 回复的消息 ID
            inline_keyboard: 内联键盘
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": content[:4096],
        }
        if msg_type == "markdown":
            payload["parse_mode"] = "MarkdownV2"
        elif msg_type == "html":
            payload["parse_mode"] = "HTML"

        if kwargs.get("reply_to"):
            payload["reply_to_message_id"] = kwargs["reply_to"]

        if kwargs.get("inline_keyboard"):
            payload["reply_markup"] = {"inline_keyboard": kwargs["inline_keyboard"]}

        try:
            resp = await self._http.post(
                f"{self.API_BASE}{self.bot_token}/sendMessage", json=payload
            )
            data = resp.json()
            return str(data.get("result", {}).get("message_id", ""))
        except Exception as e:
            await self._emit("error", platform="telegram", error=str(e))
            return ""

    async def listen(self) -> None:
        """长轮询模式 — 持续获取新消息。"""
        if not self._connected:
            await self.connect()

        while self._connected:
            try:
                resp = await self._http.get(
                    f"{self.API_BASE}{self.bot_token}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 30,
                    },
                )
                data = resp.json()
                if data.get("ok"):
                    for update in data["result"]:
                        self._last_update_id = update["update_id"]
                        msg = update.get("message", {})
                        if "text" in msg:
                            await self._emit(
                                "message",
                                chat_id=str(msg["chat"]["id"]),
                                text=msg["text"],
                                user_name=msg.get("from", {}).get("username", ""),
                            )
            except Exception:
                await asyncio.sleep(3)
