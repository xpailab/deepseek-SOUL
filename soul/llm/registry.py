"""LLM 适配器注册中心 — 自动发现与管理适配器。

支持动态注册、自动检测、速率限制统一管理。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from soul.llm.base import BaseAdapter, LLMResponse
from soul.llm.claude import ClaudeAdapter
from soul.llm.deepseek import DeepSeekAdapter
from soul.llm.openai import OpenAIAdapter
from soul.types import LLMConfig, Message


class RateLimiter:
    """基于令牌桶的速率限制器。"""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._tokens = max_requests
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self.max_requests,
                self._tokens + elapsed * (self.max_requests / self.window),
            )
            self._last_refill = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    async def wait_and_acquire(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.acquire():
                return True
            await asyncio.sleep(0.5)
        return False


class AdapterRegistry:
    """适配器注册中心。

    自动发现可用适配器，管理生命周期，统一速率限制。
    """

    _BUILTIN: dict[str, type[BaseAdapter]] = {
        "deepseek": DeepSeekAdapter,
        "claude": ClaudeAdapter,
        "openai": OpenAIAdapter,
    }

    def __init__(self):
        self._adapters: dict[str, BaseAdapter] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}

    def register(self, name: str, adapter_cls: type[BaseAdapter]) -> None:
        """注册自定义适配器。"""
        self._BUILTIN[name] = adapter_cls

    def get(self, config: LLMConfig | None = None, provider: str = "") -> BaseAdapter:
        """获取或创建适配器实例。"""
        provider_name = provider or config.provider if config else "deepseek"
        cache_key = f"{provider_name}:{config.model if config else 'default'}"

        if cache_key not in self._adapters:
            if config is None:
                config = LLMConfig(provider=provider_name)
            adapter_cls = self._BUILTIN.get(provider_name)
            if adapter_cls is None:
                # 回退到 DeepSeek（兼容 OpenAI API 格式）
                adapter_cls = DeepSeekAdapter
            self._adapters[cache_key] = adapter_cls(config)

        return self._adapters[cache_key]

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        config: LLMConfig | None = None,
        provider: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        adapter = self.get(config, provider)
        rl = self._get_rate_limiter(adapter.provider_name)
        if not await rl.wait_and_acquire():
            return LLMResponse(
                content="Rate limit exceeded. Please wait and try again.",
                finish_reason="error",
            )
        return await adapter.chat(messages, tools, system_prompt, **kwargs)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        config: LLMConfig | None = None,
        provider: str = "",
        **kwargs: Any,
    ):
        adapter = self.get(config, provider)
        async for chunk in adapter.chat_stream(messages, tools, system_prompt, **kwargs):
            yield chunk

    def supports_tools(self, provider: str = "deepseek") -> bool:
        adapter = self._adapters.get(provider)
        if adapter:
            return adapter.supports_tools()
        return True

    def _get_rate_limiter(self, provider: str) -> RateLimiter:
        if provider not in self._rate_limiters:
            self._rate_limiters[provider] = RateLimiter()
        return self._rate_limiters[provider]

    async def close_all(self) -> None:
        for adapter in self._adapters.values():
            if hasattr(adapter, "close"):
                await adapter.close()
        self._adapters.clear()

    @property
    def available_providers(self) -> list[str]:
        return list(self._BUILTIN.keys())
