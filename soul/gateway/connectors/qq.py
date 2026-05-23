"""QQ 连接器 — 支持 QQ 开放平台 Bot API。

接入方式:
1. QQ 开放平台 (https://q.qq.com) 创建机器人
2. 获取 BotAppID + BotToken
3. 支持: 文字消息 / Markdown 消息 / 图片消息
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from soul.gateway.session_sync import PlatformConnector


class QQConnector(PlatformConnector):
    """QQ 机器人连接器。

    使用:
        qq = QQConnector(bot_app_id="xxx", bot_token="xxx")
        await qq.connect()
        await qq.send_message("你好", chat_id="user_openid")

    支持的消息类型:
    - text: 普通文字
    - markdown: Markdown 格式
    - image: 图片消息
    """

    API_BASE = "https://api.sgroup.qq.com"

    def __init__(
        self,
        bot_app_id: str = "",
        bot_token: str = "",
        client_secret: str = "",
        config: dict[str, Any] | None = None,
    ):
        super().__init__("qq", config)
        self.bot_app_id = bot_app_id
        self.bot_token = bot_token
        self.client_secret = client_secret
        self._access_token: str = ""
        self._token_expires_at: float = 0.0
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """建立连接 — 获取 access_token。"""
        try:
            self._http = httpx.AsyncClient(timeout=30.0)
            await self._refresh_token()
            self._connected = True
            await self._emit("connect", platform="qq", status="connected")
            return True
        except Exception as e:
            await self._emit("error", platform="qq", error=str(e))
            return False

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False

    async def send_message(
        self, content: str, chat_id: str, msg_type: str = "text", **kwargs
    ) -> str:
        """发送消息到 QQ。

        Args:
            content: 消息内容
            chat_id: 接收者 openid / group_openid
            msg_type: text / markdown / image
        """
        if not self._connected:
            await self.connect()

        await self._ensure_token()

        if msg_type == "markdown":
            payload = self._build_markdown(content, chat_id)
        elif msg_type == "image":
            payload = self._build_image(content, chat_id)
        else:
            payload = self._build_text(content, chat_id)

        try:
            resp = await self._http.post(
                f"{self.API_BASE}/v2/users/{chat_id}/messages",
                json=payload,
                headers={
                    "Authorization": f"QQBot {self._access_token}",
                    "X-Union-Appid": self.bot_app_id,
                },
            )
            data = resp.json()
            return data.get("id", "")
        except Exception as e:
            await self._emit("error", platform="qq", error=str(e))
            return ""

    async def listen(self) -> None:
        """QQ 使用 Webhook 回调，不适用长轮询。需在开放平台配置回调 URL。"""
        pass

    # ── 消息构建 ──

    def _build_text(self, content: str, chat_id: str) -> dict:
        return {
            "content": content[:2000],
            "msg_type": 0,
            "msg_id": f"msg_{int(time.time()*1000)}",
        }

    def _build_markdown(self, content: str, chat_id: str) -> dict:
        params = []
        # 简化 Markdown 转 QQ 模板格式
        for line in content.split("\n")[:50]:
            line = line.strip()
            if not line:
                continue
            params.append(line)
        template = "\n".join(params)
        return {
            "markdown": {
                "content": template,
            },
            "msg_type": 2,
            "msg_id": f"md_{int(time.time()*1000)}",
        }

    def _build_image(self, media_url: str, chat_id: str) -> dict:
        return {
            "image": media_url,
            "msg_type": 3,
            "msg_id": f"img_{int(time.time()*1000)}",
        }

    # ── Token 管理 ──

    async def _refresh_token(self) -> None:
        """获取/刷新 access_token。"""
        resp = await self._http.post(
            "https://bots.qq.com/app/getAppAccessToken",
            json={
                "appId": self.bot_app_id,
                "clientSecret": self.client_secret,
            },
        )
        data = resp.json()
        self._access_token = data.get("access_token", "")
        expires_in = data.get("expires_in", 7200)
        self._token_expires_at = time.time() + expires_in - 300

    async def _ensure_token(self) -> None:
        if time.time() > self._token_expires_at:
            await self._refresh_token()
