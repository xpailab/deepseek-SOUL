"""LLM 适配器系统测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from soul.types import LLMConfig


class TestAdapterRegistry:
    def test_register_and_get(self):
        from soul.llm.registry import AdapterRegistry
        reg = AdapterRegistry()
        adapter = reg.get(LLMConfig(provider="deepseek", model="deepseek-chat"))
        assert adapter is not None

    def test_available_providers(self):
        from soul.llm.registry import AdapterRegistry
        reg = AdapterRegistry()
        providers = reg.available_providers
        assert "deepseek" in providers

    def test_unknown_provider_falls_back(self):
        from soul.llm.registry import AdapterRegistry
        reg = AdapterRegistry()
        adapter = reg.get(LLMConfig(provider="unknown_provider", model="test"))
        assert adapter is not None  # 回退到 DeepSeek


class TestLLMResponse:
    def test_response_creation(self):
        from soul.llm.base import LLMResponse
        resp = LLMResponse(
            content="hello",
            tool_calls=[],
            finish_reason="stop",
            reasoning_content="",
        )
        assert resp.content == "hello"
        assert resp.tool_calls == []
        assert resp.finish_reason == "stop"


class TestDeepSeekAdapter:
    def test_messages_to_api_format(self):
        from soul.llm.deepseek import DeepSeekAdapter
        from soul.types import Message, MessageRole
        adapter = DeepSeekAdapter(LLMConfig(provider="deepseek", model="deepseek-chat"))
        msgs = [
            Message(role=MessageRole.USER, content="hello"),
        ]
        formatted = adapter._messages_to_api_format(msgs)
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "hello"

    def test_supports_tools(self):
        from soul.llm.deepseek import DeepSeekAdapter
        adapter = DeepSeekAdapter(LLMConfig(provider="deepseek", model="deepseek-chat"))
        assert adapter.supports_tools() is True
