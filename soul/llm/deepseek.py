"""DeepSeek API 适配器。

DeepSeek API 兼容 OpenAI 格式，所以此适配器
复用大量 OpenAI 标准模式。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx

from soul.llm.base import BaseAdapter, LLMResponse
from soul.types import LLMConfig, Message, StreamChunk, ToolCall


class DeepSeekAdapter(BaseAdapter):
    """DeepSeek API 适配器。

    API 端点: https://api.deepseek.com/v1
    兼容 OpenAI Chat Completions API 格式。
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        if not config.api_base:
            config.api_base = "https://api.deepseek.com/v1"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    def supports_tools(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        start = time.time()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_to_api_format(messages, system_prompt),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False,
        }
        if tools:
            payload["tools"] = self._tools_to_api_format(tools)

        for attempt in range(self.config.max_retries):
            try:
                resp = await client.post("/chat/completions", json=payload)
                if resp.status_code >= 400:
                    body = resp.text[:2000]
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {body}",
                        request=resp.request,
                        response=resp,
                    )
                data = resp.json()
                break
            except httpx.HTTPStatusError as e:
                if attempt == self.config.max_retries - 1:
                    return LLMResponse(
                        content=f"API Error: {e.response.status_code}\n{e.response.text[:500] if hasattr(e, 'response') else str(e)}",
                        finish_reason="error",
                        duration_ms=(time.time() - start) * 1000,
                    )
                await asyncio.sleep(2 ** attempt)
            except httpx.RequestError:
                if attempt == self.config.max_retries - 1:
                    raise

        choice = data["choices"][0]
        content = choice["message"].get("content", "") or ""
        reasoning = choice["message"].get("reasoning_content", "") or ""
        usage = data.get("usage", {})

        tool_calls = []
        if "tool_calls" in choice["message"]:
            for tc in choice["message"]["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=args,
                ))

        self._total_tokens += usage.get("total_tokens", 0)
        self._total_requests += 1

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            reasoning_content=reasoning,
            duration_ms=(time.time() - start) * 1000,
            model=data.get("model", self.config.model),
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_to_api_format(messages, system_prompt),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }
        if tools:
            payload["tools"] = self._tools_to_api_format(tools)

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            tool_call_buf: dict[int, dict[str, Any]] = {}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    yield StreamChunk(finish_reason="stop")
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})

                    if "content" in delta and delta["content"]:
                        yield StreamChunk(content=delta["content"])

                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        yield StreamChunk(reasoning_content=delta["reasoning_content"])

                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_call_buf:
                                tool_call_buf[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            if "id" in tc:
                                tool_call_buf[idx]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_call_buf[idx]["name"] += tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_call_buf[idx]["arguments"] += tc["function"]["arguments"]

                    choice = data["choices"][0]
                    if "finish_reason" in choice:
                        if choice.get("finish_reason"):
                            for idx, buf in tool_call_buf.items():
                                try:
                                    args = json.loads(buf["arguments"])
                                except json.JSONDecodeError:
                                    args = {"raw": buf["arguments"]}
                                yield StreamChunk(
                                    tool_call=ToolCall(
                                        id=buf["id"],
                                        name=buf["name"],
                                        arguments=args,
                                    )
                                )
                            yield StreamChunk(
                                finish_reason=choice["finish_reason"],
                                usage=data.get("usage"),
                            )

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        # 确保流结束时总是发送finish标记
        yield StreamChunk(finish_reason="stop")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
