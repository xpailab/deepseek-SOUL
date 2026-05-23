"""钉钉连接器 — 支持群机器人 Webhook + 企业应用。

接入方式:
1. 群机器人: 钉钉群 → 设置 → 智能群助手 → 添加机器人 → Webhook URL
2. 企业应用: 钉钉开放平台 → 创建应用 → AppKey + AppSecret

消息类型:
- text: 纯文本
- markdown: Markdown 格式（支持标题/加粗/链接/图片）
- actionCard: 交互卡片（按钮跳转）
- feedCard: 多图文卡片
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from soul.gateway.session_sync import PlatformConnector


class DingTalkConnector(PlatformConnector):
    """钉钉连接器 — 群机器人 + 企业应用。

    使用:
        # 群机器人模式 (最简单)
        dt = DingTalkConnector(webhook_url="https://oapi.dingtalk.com/robot/send?...")
        await dt.connect()
        await dt.send_message("部署完成，请检查", msg_type="text")

        # 企业应用模式
        dt = DingTalkConnector(
            app_key="dingxxx", app_secret="secret",
            mode="app"
        )
        await dt.connect()
        await dt.send_message("你好", chat_id="manager123")

    支持消息类型:
    - text: 纯文本 @指定人
    - markdown: 标题/列表/链接/图片
    - actionCard: 带按钮的卡片
    - feedCard: 多图文链接卡片
    """

    API_BASE = "https://oapi.dingtalk.com"
    API_BASE_V2 = "https://api.dingtalk.com"

    def __init__(
        self,
        webhook_url: str = "",
        app_key: str = "",
        app_secret: str = "",
        mode: str = "webhook",  # webhook / app
        config: dict[str, Any] | None = None,
    ):
        super().__init__("dingtalk", config)
        self.webhook_url = webhook_url
        self.app_key = app_key
        self.app_secret = app_secret
        self.mode = mode
        self._access_token: str = ""
        self._token_expires_at: float = 0.0
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        try:
            self._http = httpx.AsyncClient(timeout=30.0)
            if self.mode == "app":
                await self._refresh_token()
            self._connected = True
            await self._emit("connect", platform="dingtalk", mode=self.mode)
            return True
        except Exception as e:
            await self._emit("error", platform="dingtalk", error=str(e))
            return False

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False

    async def send_message(
        self, content: str, chat_id: str = "", msg_type: str = "text", **kwargs
    ) -> str:
        """发送消息到钉钉。

        Args:
            content: 消息内容 / Markdown / 卡片 JSON
            chat_id: 企业应用模式下的 userid
            msg_type: text / markdown / actionCard / feedCard
            at_mobiles: 要 @ 的手机号列表
            at_all: 是否 @ 所有人
        """
        at_mobiles = kwargs.get("at_mobiles", [])
        at_all = kwargs.get("at_all", False)
        title = kwargs.get("title", "通知")

        if self.mode == "webhook":
            return await self._send_webhook(content, msg_type, title, at_mobiles, at_all)
        else:
            return await self._send_app(content, chat_id, msg_type)

    async def listen(self) -> None:
        """钉钉使用注册回调 URL，不能主动 listen。"""
        pass

    # ── Webhook 发送 ──

    async def _send_webhook(
        self, content: str, msg_type: str, title: str,
        at_mobiles: list[str], at_all: bool,
    ) -> str:
        at_part = {"atMobiles": at_mobiles, "isAtAll": at_all}

        if msg_type == "markdown":
            payload = {
                "msgtype": "markdown",
                "markdown": {"title": title[:50], "text": content},
                "at": at_part,
            }
        elif msg_type == "actionCard":
            # content should be actionCard JSON
            payload = {"msgtype": "actionCard", "actionCard": json.loads(content) if isinstance(content, str) else content}
        elif msg_type == "feedCard":
            payload = {"msgtype": "feedCard", "feedCard": json.loads(content) if isinstance(content, str) else content}
        else:
            payload = {
                "msgtype": "text",
                "text": {"content": content[:20000]},
                "at": at_part,
            }

        try:
            import json as _json
            resp = await self._http.post(self.webhook_url, json=payload)
            data = resp.json()
            return str(data.get("errcode", -1))
        except Exception as e:
            await self._emit("error", platform="dingtalk", error=str(e))
            return ""

    # ── 企业应用发送 ──

    async def _send_app(self, content: str, userid: str, msg_type: str) -> str:
        await self._ensure_token()
        payload = {
            "agent_id": self.app_key,
            "userid_list": userid,
            "msg": {
                "msgtype": msg_type,
                msg_type: {"content": content[:20000]},
            },
        }
        try:
            resp = await self._http.post(
                f"{self.API_BASE}/topapi/message/corpconversation/asyncsend_v2?access_token={self._access_token}",
                json=payload,
            )
            data = resp.json()
            return str(data.get("task_id", ""))
        except Exception as e:
            await self._emit("error", platform="dingtalk", error=str(e))
            return ""

    # ── Token ──

    async def _refresh_token(self) -> None:
        resp = await self._http.get(
            f"{self.API_BASE}/gettoken",
            params={"appkey": self.app_key, "appsecret": self.app_secret},
        )
        data = resp.json()
        self._access_token = data.get("access_token", "")
        expires_in = data.get("expires_in", 7200)
        self._token_expires_at = time.time() + expires_in - 300

    async def _ensure_token(self) -> None:
        if time.time() > self._token_expires_at:
            await self._refresh_token()
