"""Claude (Anthropic) API 适配器。

使用 Anthropic Messages API.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx

from soul.llm.base import BaseAdapter, LLMResponse
from soul.types import LLMConfig, Message, MessageRole, StreamChunk, ToolCall


class ClaudeAdapter(BaseAdapter):
    """Claude API 适配器。

    API 端点: https://api.anthropic.com/v1
    """

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        if not config.api_base:
            config.api_base = "https://api.anthropic.com/v1"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": self.ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    def supports_tools(self) -> bool:
        return True

    def _to_claude_messages(
        self, messages: list[Message], system_prompt: str = ""
    ) -> tuple[str, list[dict[str, Any]]]:
        """转为 Claude 消息格式。返回 (system, messages)。"""
        claude_msgs: list[dict[str, Any]] = []
        for msg in messages:
            role = "assistant" if msg.role == MessageRole.ASSISTANT else "user"

            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })

            if msg.tool_results:
                for tr in msg.tool_results:
                    claude_msgs.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tr.call_id,
                            "content": str(tr.result) if tr.success else tr.error or "",
                        }],
                    })
                continue

            claude_msgs.append({"role": role, "content": content})

        return system_prompt, claude_msgs

    def _to_claude_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": t.get("parameters", {}).get("properties", {}),
                    "required": t.get("parameters", {}).get("required", []),
                },
            }
            for t in tools
        ]

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        start = time.time()
        system, claude_msgs = self._to_claude_messages(messages, system_prompt)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": claude_msgs,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._to_claude_tools(tools)

        for attempt in range(self.config.max_retries):
            try:
                resp = await client.post("/messages", json=payload)
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.HTTPStatusError as e:
                if attempt == self.config.max_retries - 1:
                    return LLMResponse(
                        content=f"API Error: {e.response.status_code}",
                        finish_reason="error",
                        duration_ms=(time.time() - start) * 1000,
                    )
                await asyncio.sleep(2 ** attempt)

        content = ""
        tool_calls: list[ToolCall] = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        usage = data.get("usage", {})
        self._total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        self._total_requests += 1

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason", "stop"),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            },
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
        system, claude_msgs = self._to_claude_messages(messages, system_prompt)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": claude_msgs,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._to_claude_tools(tools)

        async with client.stream("POST", "/messages", json=payload) as resp:
            resp.raise_for_status()
            tool_buf: dict[str, dict[str, Any]] = {}

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                evt_type = event.get("type", "")

                if evt_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(content=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        pass  # 累积处理

                elif evt_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_buf[block["id"]] = {
                            "id": block["id"],
                            "name": block.get("name", ""),
                            "input": "",
                        }

                elif evt_type == "content_block_stop":
                    pass

                elif evt_type == "message_delta":
                    usage = event.get("usage", {})
                    for tid, buf in tool_buf.items():
                        try:
                            args = json.loads(buf["input"]) if buf["input"] else {}
                        except json.JSONDecodeError:
                            args = {"raw": buf["input"]}
                        yield StreamChunk(
                            tool_call=ToolCall(
                                id=buf["id"],
                                name=buf["name"],
                                arguments=args,
                            )
                        )
                    yield StreamChunk(
                        finish_reason=event.get("delta", {}).get("stop_reason", "stop"),
                        usage=usage,
                    )

                elif evt_type == "message_stop":
                    break

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
