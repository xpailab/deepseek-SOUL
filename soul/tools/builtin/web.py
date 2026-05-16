"""Web 工具 — HTTP 请求和网页抓取。"""

from __future__ import annotations

from typing import Any

import httpx

from soul.tools.registry import ToolDef
from soul.types import ToolRisk


class WebTool:
    """HTTP 请求和网页内容获取工具。"""

    NAME = "web"
    DESCRIPTION = "发送 HTTP 请求获取网页内容或 API 数据。"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["fetch", "search"],
                "description": "操作类型: fetch (获取URL内容), search (搜索网页)",
            },
            "url": {
                "type": "string",
                "description": "目标 URL（fetch 操作）",
            },
            "query": {
                "type": "string",
                "description": "搜索关键词（search 操作）",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST"],
                "description": "HTTP 方法",
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "description": "自定义 HTTP 头",
            },
            "body": {
                "type": "string",
                "description": "请求体（POST 操作）",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "DeepSoul-Agent/0.1 (+https://github.com/deepseek-SOUL)",
                },
            )
        return self._client

    async def execute(
        self,
        action: str,
        url: str = "",
        query: str = "",
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str = "",
    ) -> dict[str, Any]:
        """执行 Web 操作。"""
        try:
            if action == "fetch":
                return await self._fetch(url, method, headers, body)
            elif action == "search":
                return await self._search(query)
            else:
                return {"error": f"未知操作: {action}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _fetch(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str = "",
    ) -> dict[str, Any]:
        client = await self._get_client()

        if method == "POST":
            resp = await client.post(url, headers=headers, content=body)
        else:
            resp = await client.get(url, headers=headers)

        content_type = resp.headers.get("content-type", "")

        result: dict[str, Any] = {
            "status_code": resp.status_code,
            "url": str(resp.url),
            "headers": dict(resp.headers),
            "success": resp.is_success,
        }

        # 只返回文本内容，忽略二进制
        if "text" in content_type or "json" in content_type or "xml" in content_type:
            text = resp.text[:50000]  # 限制 50KB
            result["content"] = text
            result["content_length"] = len(text)
            result["truncated"] = len(resp.text) > 50000
        else:
            result["content"] = f"[非文本内容: {content_type}, 大小: {len(resp.content)} bytes]"

        return result

    async def _search(self, query: str) -> dict[str, Any]:
        """简单的网页搜索（使用 DuckDuckGo HTML）。"""
        if not query:
            return {"error": "搜索关键词不能为空", "success": False}

        client = await self._get_client()
        # DuckDuckGo HTML 搜索（非 API，无需密钥）
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )

        if not resp.is_success:
            return {"error": f"搜索失败: HTTP {resp.status_code}", "success": False}

        # 简单提取搜索结果
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results: list[dict[str, str]] = []
                self._in_result = False
                self._in_link = False
                self._in_snippet = False
                self._current: dict[str, str] = {}
                self._data = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self._in_link = True
                    self._current = {"url": attrs_dict.get("href", "")}
                elif tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self._in_snippet = True

            def handle_data(self, data):
                if self._in_link:
                    self._current["title"] = (self._current.get("title", "") + data).strip()
                elif self._in_snippet:
                    self._current["snippet"] = (self._current.get("snippet", "") + data).strip()

            def handle_endtag(self, tag):
                if self._in_link and tag == "a":
                    self._in_link = False
                    if self._current.get("title"):
                        self.results.append(self._current)
                        self._current = {}
                elif self._in_snippet and tag == "a":
                    self._in_snippet = False

        parser = DDGParser()
        parser.feed(resp.text)

        return {
            "query": query,
            "results": parser.results[:10],
            "total": len(parser.results),
            "success": True,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @classmethod
    def to_tool_def(cls) -> ToolDef:
        return ToolDef(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            handler=cls().execute,
            parameters=cls.PARAMETERS,
            risk=ToolRisk.LOW,
            requires_approval=False,
            timeout_seconds=30,
            max_retries=2,
            tags=["web", "http", "search", "fetch"],
        )
