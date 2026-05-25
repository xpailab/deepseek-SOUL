"""微信连接器 — 支持公众号 + 企业微信。

接入方式:
1. 公众号: 微信公众平台 → 基本配置 → AppID + AppSecret
2. 企业微信: 企业微信管理后台 → 应用管理 → AgentID + Secret

消息类型: 文本 / 图文 / 模板消息
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from soul.gateway.connectors.base import PlatformConnector


class WeChatConnector(PlatformConnector):
    """微信连接器 — 公众号 + 企业微信。

    使用:
        # 公众号模式
        wx = WeChatConnector(app_id="wxAPPID", app_secret="secret", mode="mp")

        # 企业微信模式
        wx = WeChatConnector(
            corpid="wwCORPID", corpsecret="secret",
            agent_id=1000001, mode="wecom"
        )
        await wx.connect()
        await wx.send_message("你好", chat_id="oUSER_OPENID")
    """

    MP_API = "https://api.weixin.qq.com/cgi-bin"
    WECOM_API = "https://qyapi.weixin.qq.com/cgi-bin"

    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        corpid: str = "",
        corpsecret: str = "",
        agent_id: int = 0,
        mode: str = "mp",  # mp=公众号, wecom=企业微信
        token: str = "",
        encoding_aes_key: str = "",
        config: dict[str, Any] | None = None,
    ):
        super().__init__("wechat", config)
        self.app_id = app_id
        self.app_secret = app_secret
        self.corpid = corpid
        self.corpsecret = corpsecret
        self.agent_id = agent_id
        self.mode = mode
        self.token = token  # 服务器配置 Token
        self.encoding_aes_key = encoding_aes_key
        self._access_token: str = ""
        self._token_expires_at: float = 0.0
        self._http: httpx.AsyncClient | None = None

    @property
    def api_base(self) -> str:
        return self.WECOM_API if self.mode == "wecom" else self.MP_API

    async def connect(self) -> bool:
        try:
            self._http = httpx.AsyncClient(timeout=30.0)
            await self._refresh_token()
            self._connected = True
            await self._emit("connect", platform="wechat", mode=self.mode)
            return True
        except Exception as e:
            await self._emit("error", platform="wechat", error=str(e))
            return False

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False

    async def send_message(
        self, content: str, chat_id: str, msg_type: str = "text", **kwargs
    ) -> str:
        """发送消息。

        Args:
            content: 消息内容
            chat_id: 公众号 openid / 企业微信 userid
            msg_type: text / image / news
        """
        await self._ensure_token()

        if self.mode == "wecom":
            payload = self._build_wecom_message(content, chat_id, msg_type)
        else:
            payload = self._build_mp_message(content, chat_id, msg_type)

        try:
            url = (
                f"{self.api_base}/message/send?access_token={self._access_token}"
                if self.mode == "mp"
                else f"{self.api_base}/message/send?access_token={self._access_token}"
            )
            resp = await self._http.post(url, json=payload)
            data = resp.json()
            if data.get("errcode") == 0:
                return data.get("msgid", str(time.time()))
            return ""
        except Exception as e:
            await self._emit("error", platform="wechat", error=str(e))
            return ""

    async def listen(self) -> None:
        """微信使用服务器回调 URL，不能主动 listen。需在公众平台配置。"""
        pass

    # ── 消息构建 ──

    def _build_mp_message(self, content: str, openid: str, msg_type: str) -> dict:
        if msg_type == "image":
            return {
                "touser": openid,
                "msgtype": "image",
                "image": {"media_id": content},
            }
        # 公众号文本消息限制 2048 字符
        return {
            "touser": openid,
            "msgtype": "text",
            "text": {"content": content[:2048]},
        }

    def _build_wecom_message(self, content: str, userid: str, msg_type: str) -> dict:
        base = {
            "touser": userid,
            "agentid": self.agent_id,
        }
        if msg_type == "markdown":
            base["msgtype"] = "markdown"
            base["markdown"] = {"content": content[:4096]}
        elif msg_type == "image":
            base["msgtype"] = "image"
            base["image"] = {"media_id": content}
        else:
            base["msgtype"] = "text"
            base["text"] = {"content": content[:2048]}
        return base

    # ── Token ──

    async def _refresh_token(self) -> None:
        if self.mode == "wecom":
            url = f"{self.WECOM_API}/gettoken?corpid={self.corpid}&corpsecret={self.corpsecret}"
        else:
            url = f"{self.MP_API}/token?grant_type=client_credential&appid={self.app_id}&secret={self.app_secret}"
        resp = await self._http.get(url)
        data = resp.json()
        self._access_token = data.get("access_token", "")
        expires_in = data.get("expires_in", 7200)
        self._token_expires_at = time.time() + expires_in - 300

    async def _ensure_token(self) -> None:
        if time.time() > self._token_expires_at:
            await self._refresh_token()
