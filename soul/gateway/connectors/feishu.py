"""飞书连接器 — 支持自定义机器人 + 企业应用。

接入方式:
1. 自定义机器人: 飞书群 → 设置 → 群机器人 → 添加机器人 → Webhook URL
   最简单，无需审批，适合通知类场景
2. 企业应用: 飞书开放平台 → 创建应用 → AppID + AppSecret
   支持交互消息、消息卡片、审批流

消息类型:
- text: 纯文本（最长 30KB）
- interactive: 消息卡片（JSON 格式，支持按钮/下拉/日期选择器）
- image: 图片消息
- share_chat: 分享群名片
- post: 富文本（支持 at、图片、超链接）
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from soul.gateway.session_sync import PlatformConnector


class FeishuConnector(PlatformConnector):
    """飞书连接器 — 自定义机器人 + 企业应用。

    使用:
        # 自定义机器人 (推荐 — 零审批)
        fs = FeishuConnector(
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
        )
        await fs.connect()
        await fs.send_message("代码审查完成，请查看")

        # 企业应用 (功能全)
        fs = FeishuConnector(
            app_id="cli_xxx", app_secret="secret",
            mode="app"
        )
        await fs.connect()
        await fs.send_message(
            content="你好",
            chat_id="ou_USERID",
            msg_type="interactive",
            card={
                "header": {"title": {"content": "审查结果"}},
                "elements": [{"tag": "div", "text": {"content": "3个问题待处理"}}],
            }
        )

    支持消息类型:
    - text: 纯文本
    - interactive: 消息卡片
    - image: 图片
    - share_chat: 分享群聊
    - post: 富文本消息
    """

    API_BASE = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        webhook_url: str = "",
        app_id: str = "",
        app_secret: str = "",
        mode: str = "webhook",  # webhook / app
        verify_token: str = "",
        encrypt_key: str = "",
        config: dict[str, Any] | None = None,
    ):
        super().__init__("feishu", config)
        self.webhook_url = webhook_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.mode = mode
        self.verify_token = verify_token
        self.encrypt_key = encrypt_key
        self._tenant_access_token: str = ""
        self._token_expires_at: float = 0.0
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        try:
            self._http = httpx.AsyncClient(timeout=30.0)
            if self.mode == "app":
                await self._refresh_token()
            self._connected = True
            await self._emit("connect", platform="feishu", mode=self.mode)
            return True
        except Exception as e:
            await self._emit("error", platform="feishu", error=str(e))
            return False

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False

    async def send_message(
        self, content: str, chat_id: str = "", msg_type: str = "text", **kwargs
    ) -> str:
        """发送消息到飞书。

        Args:
            content: 消息内容
            chat_id: 企业应用模式下的 open_id / chat_id / user_id
            msg_type: text / interactive / image / post / share_chat
            card: 消息卡片 dict（msg_type=interactive 时）
        """
        card = kwargs.get("card")

        if self.mode == "webhook":
            return await self._send_webhook(content, msg_type, card)
        else:
            return await self._send_app(content, chat_id, msg_type, card)

    async def listen(self) -> None:
        """飞书使用事件订阅 URL，不能主动 listen。"""
        pass

    # ── Webhook 发送 ──

    async def _send_webhook(
        self, content: str, msg_type: str, card: dict | None
    ) -> str:
        if msg_type == "interactive" and card:
            payload = {"msg_type": "interactive", "card": card}
        else:
            payload = {
                "msg_type": "text",
                "content": {"text": content[:30720]},
            }

        try:
            resp = await self._http.post(self.webhook_url, json=payload)
            data = resp.json()
            return data.get("StatusMessage", "")
        except Exception as e:
            await self._emit("error", platform="feishu", error=str(e))
            return ""

    # ── 企业应用发送 ──

    async def _send_app(
        self, content: str, receive_id: str, msg_type: str, card: dict | None
    ) -> str:
        await self._ensure_token()

        if msg_type == "interactive" and card:
            content_payload = card
            content_key = "card"
        elif msg_type == "post":
            content_payload = {
                "zh_cn": {
                    "title": card.get("title", "") if card else "",
                    "content": [[{"tag": "text", "text": content[:30720]}]] if content else [],
                }
            }
            content_key = "content"
        else:
            content_payload = {"text": content[:30720]}
            content_key = "content"

        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": content_payload if content_key == "content" else None,
            "card": card if msg_type == "interactive" else None,
        }

        try:
            resp = await self._http.post(
                f"{self.API_BASE}/im/v1/messages?receive_id_type=open_id",
                json=payload,
                headers={"Authorization": f"Bearer {self._tenant_access_token}"},
            )
            data = resp.json()
            return data.get("data", {}).get("message_id", "")
        except Exception as e:
            await self._emit("error", platform="feishu", error=str(e))
            return ""

    # ── Token ──

    async def _refresh_token(self) -> None:
        resp = await self._http.post(
            f"{self.API_BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        data = resp.json()
        self._tenant_access_token = data.get("tenant_access_token", "")
        expires_in = data.get("expire", 7200)
        self._token_expires_at = time.time() + expires_in - 300

    async def _ensure_token(self) -> None:
        if time.time() > self._token_expires_at:
            await self._refresh_token()
