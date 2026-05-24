"""网关路由器测试：MessageRouter, ChannelMessage。"""

from __future__ import annotations

import pytest

from soul.gateway.router import ChannelMessage, MessageRouter
from soul.types import MessageRole, QueueMode


class TestChannelMessage:
    def test_creation(self):
        msg = ChannelMessage(raw_text="你好", channel="cli", channel_user_id="user1")
        assert msg.raw_text == "你好"
        assert msg.channel == "cli"
        assert msg.channel_user_id == "user1"
        assert msg.id.startswith("chmsg_")
        assert msg.session_id == "cli:user1"

    def test_to_message(self):
        msg = ChannelMessage(raw_text="测试消息", channel="telegram", channel_user_id="tg_123")
        m = msg.to_message()
        assert m.role == MessageRole.USER
        assert m.content == "测试消息"
        assert m.metadata["channel"] == "telegram"

    def test_resolve_queue_mode_interrupt(self):
        msg = ChannelMessage(raw_text="stop", channel="cli", channel_user_id="u1")
        assert msg.resolve_queue_mode("stop") == QueueMode.INTERRUPT
        assert msg.resolve_queue_mode("取消") == QueueMode.INTERRUPT
        assert msg.resolve_queue_mode("abort") == QueueMode.INTERRUPT

    def test_resolve_queue_mode_steer(self):
        msg = ChannelMessage(raw_text="", channel="cli", channel_user_id="u1")
        assert msg.resolve_queue_mode("用中文回复") == QueueMode.STEER
        assert msg.resolve_queue_mode("不要用表格") == QueueMode.STEER

    def test_resolve_queue_mode_default(self):
        msg = ChannelMessage(raw_text="", channel="cli", channel_user_id="u1")
        assert msg.resolve_queue_mode("帮我创建一个Python项目") == QueueMode.ADAPTIVE


class TestMessageRouter:
    @pytest.mark.asyncio
    async def test_register_and_route(self):
        router = MessageRouter()

        async def handler(msg):
            return {"status": "ok", "text": msg.raw_text}

        router.register_handler("telegram", handler)
        assert "telegram" in router._handlers

        msg = ChannelMessage(raw_text="hello", channel="telegram", channel_user_id="u1")
        result = await router.route(msg)
        assert result["status"] == "ok"
        assert result["text"] == "hello"

    @pytest.mark.asyncio
    async def test_route_unknown_channel(self):
        router = MessageRouter()
        msg = ChannelMessage(raw_text="hi", channel="unknown", channel_user_id="u1")
        result = await router.route(msg)
        assert "error" in result

    def test_get_session_id(self):
        router = MessageRouter()
        msg = ChannelMessage(raw_text="hi", channel="cli", channel_user_id="u1")
        assert router.get_session_id(msg) == "cli:u1"
