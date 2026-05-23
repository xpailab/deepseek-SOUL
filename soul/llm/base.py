"""LLM 适配器基类 — 定义统一接口。

所有 LLM 提供商适配器都继承此基类，确保接口一致。
"""

from __future__ import annotations

import asyncio
import time
import json
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
    usage: dict[str, Any] = field(default_factory=dict)  # prompt_tokens, completion_tokens (+ nested details)
    reasoning_content: str = ""  # DeepSeek 思考模式的内部推理，发回 API 时必须原样保留
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
            # TOOL 角色消息直接转换
            if msg.role.value == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.metadata.get("tool_call_id", ""),
                    "content": msg.content,
                })
                continue

            api_msg: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.tool_calls:
                api_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in msg.tool_calls
                ]
            # DeepSeek 思考模式：reasoning_content 必须原样传回
            if msg.reasoning_content:
                api_msg["reasoning_content"] = msg.reasoning_content
            # tool_results 仅用于内部追踪，不拼到 API 格式（由独立的 TOOL 消息承载）
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
