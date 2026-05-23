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
        # SSRF 防护：验证 URL 协议和主机
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {"error": f"不支持的协议: {parsed.scheme}", "success": False}
        # 阻止内网地址
        blocked_hosts = ("localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "metadata.google.internal")
        hostname = (parsed.hostname or "").lower()
        if hostname in blocked_hosts or hostname.startswith("10.") or hostname.startswith("192.168.") or hostname.startswith("172.16."):
            return {"error": "不允许访问内网地址", "success": False}

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
        """多引擎网页搜索 — 自动选择可用引擎。

        优先级: Bing → DuckDuckGo → 百度
        一个引擎不可用时自动尝试下一个。
        """
        if not query:
            return {"error": "搜索关键词不能为空", "success": False}

        client = await self._get_client()

        # 按优先级尝试各引擎
        backends = [
            ("bing", self._search_bing),
            ("ddg", self._search_ddg),
            ("baidu", self._search_baidu),
            ("sogou", self._search_sogou),
        ]

        for name, handler in backends:
            try:
                result = await handler(client, query)
                if result["results"] or result.get("success"):
                    result["engine"] = name
                    return result
            except Exception:
                continue

        return {"error": "所有搜索引擎均不可用", "success": False, "results": []}

    # ── Bing 搜索 ──

    async def _search_bing(self, client: httpx.AsyncClient, query: str) -> dict[str, Any]:
        """Bing 搜索 — 全球可用，中国可访问。"""
        resp = await client.get(
            "https://www.bing.com/search",
            params={"q": query, "setlang": "zh-Hans"},
            headers={"Accept-Language": "zh-CN,zh;q=0.9"},
        )
        if not resp.is_success:
            return {"success": False, "results": []}

        import re
        results: list[dict[str, str]] = []
        # Bing 结果在 <li class="b_algo"> 中
        blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', resp.text, re.DOTALL)
        for block in blocks[:10]:
            link = re.search(r'<a[^>]*href="(https?://[^"]+)"', block)
            title = re.search(r'<a[^>]*>(.*?)</a>', block, re.DOTALL)
            snippet = re.search(r'(?:<p[^>]*>|<div class="b_caption[^"]*"[^>]*>)(.*?)(?:</p>|</div>)', block, re.DOTALL)
            if link and title:
                results.append({
                    "url": link.group(1),
                    "title": re.sub(r'<[^>]+>', '', title.group(1)).strip(),
                    "snippet": re.sub(r'<[^>]+>', '', snippet.group(1)).strip() if snippet else "",
                })

        return {"query": query, "results": results, "total": len(results), "success": True}

    # ── DuckDuckGo 搜索 ──

    async def _search_ddg(self, client: httpx.AsyncClient, query: str) -> dict[str, Any]:
        """DuckDuckGo HTML 搜索（海外首选，无需 API 密钥）。"""
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )
        if not resp.is_success:
            return {"success": False, "results": []}

        from html.parser import HTMLParser
        import re as _re

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results: list[dict[str, str]] = []
                self._in_link = False
                self._in_snippet = False
                self._current: dict[str, str] = {}

            def handle_starttag(self, tag, attrs):
                d = dict(attrs)
                if tag == "a" and "result__a" in d.get("class", ""):
                    self._in_link = True
                    self._current = {"url": _re.sub(r'^/+l/?\?uddg=', '', d.get("href", "")).replace("%3A%2F%2F", "://").replace("%2F", "/")}
                elif tag == "a" and "result__snippet" in d.get("class", ""):
                    self._in_snippet = True

            def handle_data(self, data):
                if self._in_link and "title" not in self._current:
                    self._current["title"] = data.strip()
                elif self._in_snippet:
                    self._current["snippet"] = (self._current.get("snippet", "") + data).strip()

            def handle_endtag(self, tag):
                if self._in_link and tag == "a":
                    self._in_link = False
                elif self._in_snippet and tag == "a":
                    self._in_snippet = False
                    if self._current.get("title"):
                        self.results.append(self._current)
                        self._current = {}

        parser = DDGParser()
        parser.feed(resp.text)
        return {"query": query, "results": parser.results[:10], "total": len(parser.results), "success": True}

    # ── 百度搜索 ──

    async def _search_baidu(self, client: httpx.AsyncClient, query: str) -> dict[str, Any]:
        """百度搜索 — 国内首选。"""
        resp = await client.get(
            "https://www.baidu.com/s",
            params={"wd": query, "rn": "10"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        )
        if not resp.is_success:
            return {"success": False, "results": []}

        import re
        results: list[dict[str, str]] = []
        # 百度结果: <div class="result c-container" ...>
        blocks = re.findall(r'<div[^>]*class="[^"]*result[^"]*c-container[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', resp.text, re.DOTALL)
        for block in blocks[:10]:
            title_m = re.search(r'<a[^>]*>(.*?)</a>', block, re.DOTALL)
            url_m = re.search(r'(?:href|data-url|mu)="(https?://[^"]+)"', block)
            snippet_m = re.search(r'<span[^>]*class="[^"]*content-right_[^"]*"[^>]*>(.*?)</span>', block, re.DOTALL)
            if not snippet_m:
                snippet_m = re.search(r'class="c-abstract"[^>]*>(.*?)</span>', block, re.DOTALL)
            if not snippet_m:
                snippet_m = re.search(r'class="c-span-last"[^>]*>(.*?)</span>', block, re.DOTALL)

            title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""
            if title and url_m:
                results.append({
                    "url": url_m.group(1),
                    "title": title,
                    "snippet": re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip()[:200] if snippet_m else "",
                })

        return {"query": query, "results": results, "total": len(results), "success": True}

    # ── 搜狗搜索 ──

    async def _search_sogou(self, client: httpx.AsyncClient, query: str) -> dict[str, Any]:
        """搜狗搜索 — 微信/知乎内容收录较好。"""
        resp = await client.get(
            "https://www.sogou.com/web",
            params={"query": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        if not resp.is_success:
            return {"success": False, "results": []}

        import re
        results: list[dict[str, str]] = []
        blocks = re.findall(r'<div[^>]*class="[^"]*rb[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', resp.text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r'<div[^>]*class="[^"]*vrwrap[^"]*"[^>]*>(.*?)</div>\s*</div>', resp.text, re.DOTALL)

        for block in blocks[:10]:
            title_m = re.search(r'<a[^>]*>(.*?)</a>', block, re.DOTALL)
            url_m = re.search(r'href="(https?://[^"]+)"', block)
            snippet_m = re.search(r'<p[^>]*class="[^"]*(?:str_info|star-wiki|space-txt)[^"]*"[^>]*>(.*?)</p>', block, re.DOTALL)

            title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""
            if title and url_m:
                results.append({
                    "url": url_m.group(1),
                    "title": title,
                    "snippet": re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip()[:200] if snippet_m else "",
                })

        return {"query": query, "results": results, "total": len(results), "success": True}

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
