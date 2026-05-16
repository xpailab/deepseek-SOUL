"""LLM 适配器基类 — 定义统一接口。

所有 LLM 提供商适配器都继承此基类，确保接口一致。
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from soul.types import LLMConfig, Message, StreamChunk, ToolCall


@dataclass
class LLMResponse:
    """LLM 返回的统一响应。"""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # stop, length, tool_calls, error
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    duration_ms: float = 0
    model: str = ""
    raw_response: Any = None


class BaseAdapter(ABC):
    """LLM 适配器基类。

    所有提供商适配器必须实现:
    - chat(): 非流式对话
    - chat_stream(): 流式对话
    - supports_tools(): 是否支持原生工具调用
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._total_tokens = 0
        self._total_requests = 0

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.replace("Adapter", "").lower()

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """非流式对话。"""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """流式对话 — 返回异步迭代器。"""
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """是否支持原生工具调用。"""
        ...

    def count_tokens(self, text: str) -> int:
        """估算 token 数。默认用字符数/4 粗略估算，子类可覆盖。"""
        return max(1, len(text) // 4)

    def get_usage_stats(self) -> dict[str, int]:
        """获取使用统计。"""
        return {
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
        }

    def _messages_to_api_format(
        self, messages: list[Message], system_prompt: str = ""
    ) -> list[dict[str, Any]]:
        """将内部消息格式转为 OpenAI API 格式。"""
        api_messages: list[dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            api_msg: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.tool_calls:
                api_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_results:
                # 工具结果作为独立消息追加
                for tr in msg.tool_results:
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tr.call_id,
                        "content": str(tr.result) if tr.success else tr.error or "",
                    })
                continue
            api_messages.append(api_msg)
        return api_messages

    def _tools_to_api_format(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """将内部工具格式转为 API 格式。"""
        if not tools:
            return None
        return [
            {"type": "function", "function": t} for t in tools
        ]
