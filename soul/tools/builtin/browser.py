"""Browser 工具 — 通过 CDP 控制 Edge/Chrome 浏览器。

无需安装额外依赖，直接用 PowerShell + CDP 协议控制浏览器。
支持: 打开页面、点击元素、输入文本、执行 JS、读取页面内容。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

from soul.tools.registry import ToolDef
from soul.types import ToolRisk

# 查找 Edge 和 PowerShell 的路径
_EDGE = (
    shutil.which("msedge")
    or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    or r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)


class BrowserTool:
    """浏览器自动化工具 — 基于 CDP (Chrome DevTools Protocol)。"""

    NAME = "browser"
    DESCRIPTION = (
        "控制浏览器(Edge/Chrome): 打开页面、hover悬停、点击网页按钮、输入文本、执行JS。"
        "hover用于触发展开下拉菜单，click用于点击。网页操作用此工具而非win工具。"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["launch", "navigate", "click", "hover", "type", "get_text", "eval"],
                "description": "操作: launch(启动浏览器), navigate(打开URL), click(点击元素), type(输入文本), get_text(获取页面文字), eval(执行JS)",
            },
            "url": {
                "type": "string",
                "description": "网页 URL（navigate 操作）",
            },
            "selector": {
                "type": "string",
                "description": "CSS 选择器或文本（click/type 操作）",
            },
            "text": {
                "type": "string",
                "description": "要输入的文本（type 操作）",
            },
            "js": {
                "type": "string",
                "description": "要执行的 JavaScript 代码（eval 操作）",
            },
        },
        "required": ["action"],
    }

    CDP_PORT = 9223

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._ws_url: str = ""

    async def _get_client(self) -> httpx.AsyncClient | None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    async def _cdp(self, method: str, params: dict | None = None) -> dict:
        """发送 CDP 命令到浏览器。"""
        if not self._ws_url:
            # 先获取 WebSocket URL
            client = await self._get_client()
            if not client:
                return {"error": "无法创建 HTTP 客户端"}
            try:
                resp = await client.get(f"http://127.0.0.1:{self.CDP_PORT}/json")
                pages = resp.json()
                if not pages:
                    return {"error": "没有打开的页面"}
                self._ws_url = pages[0].get("webSocketDebuggerUrl", "")
                if not self._ws_url:
                    return {"error": "无法获取 WebSocket URL"}
            except Exception as e:
                return {"error": f"CDP 连接失败: {e}。请确认浏览器已用 --remote-debugging-port={self.CDP_PORT} 启动"}

        # 通过 WebSocket 发送 CDP 命令
        # Python 的 websockets 库可能未安装，用 HTTP 回退
        try:
            import websockets
            payload = json.dumps({"id": 1, "method": method, "params": params or {}})
            async with websockets.connect(self._ws_url) as ws:
                await ws.send(payload)
                result = await asyncio.wait_for(ws.recv(), timeout=10)
                return json.loads(result)
        except ImportError:
            return {"error": "websockets 库未安装，无法发送 CDP 命令"}
        except Exception as e:
            return {"error": f"CDP 命令失败: {e}"}

    async def execute(
        self,
        action: str,
        url: str = "",
        selector: str = "",
        text: str = "",
        js: str = "",
    ) -> dict[str, Any]:
        try:
            if action == "launch":
                return await self._launch()
            elif action == "navigate":
                return await self._navigate(url)
            elif action == "click":
                return await self._click_element(selector)
            elif action == "hover":
                return await self._hover(selector)
            elif action == "type":
                return await self._type_text(selector, text)
            elif action == "get_text":
                return await self._get_text()
            elif action == "eval":
                return await self._eval(js)
            else:
                return {"error": f"未知操作: {action}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _launch(self) -> dict[str, Any]:
        """启动 Edge 浏览器（带调试端口）。"""
        client = await self._get_client()
        if client:
            try:
                resp = await client.get(f"http://127.0.0.1:{self.CDP_PORT}/json")
                if resp.status_code == 200:
                    return {"success": True, "stdout": "浏览器已在运行", "status": "already_running"}
            except Exception:
                pass

        # 用独立用户目录启动 Edge（不影响已运行的 Edge）
        profile = os.path.join(os.path.expanduser("~"), ".soul", "edge-profile")
        subprocess.Popen(
            [_EDGE, f"--remote-debugging-port={self.CDP_PORT}",
             f"--user-data-dir={profile}",
             "--no-first-run", "--no-default-browser-check", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        await asyncio.sleep(2)
        return {"success": True, "stdout": "浏览器已启动", "status": "launched"}

    async def _navigate(self, url: str) -> dict[str, Any]:
        """导航到指定 URL。"""
        if not url:
            return {"error": "URL 不能为空", "success": False}
        result = await self._cdp("Page.navigate", {"url": url})
        if result.get("error"):
            # CDP 失败 → 用 bash 回退
            return await self._fallback_navigate(url)
        return {"success": True, "stdout": f"已导航到 {url}"}

    async def _hover(self, selector: str) -> dict[str, Any]:
        """鼠标悬停 — 移动光标到元素中心触发真实 hover 效果。"""
        if not selector:
            return {"error": "选择器不能为空", "success": False}
        js_code = f"""
        (function() {{
            let el = document.querySelector('{selector}');
            if (!el) {{
                let all = document.querySelectorAll('a,button,span,div,li');
                for (let e of all) {{
                    if (e.textContent.trim().indexOf('{selector}') >= 0 && e.offsetWidth > 0) {{
                        el = e; break;
                    }}
                }}
            }}
            if (!el) return 'not_found';
            let r = el.getBoundingClientRect();
            return JSON.stringify({{x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2), found: true}});
        }})()
        """
        result = await self._cdp("Runtime.evaluate", {"expression": js_code, "returnByValue": True})
        if result.get("error"): return result
        value = result.get("result", {}).get("result", {}).get("value", "")
        try:
            pos = json.loads(value)
            if pos.get("found"):
                # 用 CDP Input.dispatchMouseEvent 移动鼠标
                mr = await self._cdp("Input.dispatchMouseEvent", {
                    "type": "mouseMoved", "x": pos["x"], "y": pos["y"]
                })
                await asyncio.sleep(0.2)
                return {"success": True, "stdout": f"hovered at ({pos['x']},{pos['y']})"}
        except Exception:
            pass
        return {"success": False, "stdout": str(value)[:100]}

    async def _click_element(self, selector: str) -> dict[str, Any]:
        """点击元素（支持 CSS 选择器和文本匹配）。"""
        if not selector:
            return {"error": "选择器不能为空", "success": False}
        # 用 JS 查找并点击
        js_code = f"""
        (function() {{
            // 先按 CSS 选择器找
            let el = document.querySelector('{selector}');
            // 没找到 → 按文本找
            if (!el) {{
                let all = document.querySelectorAll('a,button,span,div,li');
                for (let e of all) {{
                    if (e.textContent.trim().indexOf('{selector}') >= 0 && e.offsetWidth > 0) {{
                        el = e; break;
                    }}
                }}
            }}
            if (el) {{ el.click(); return 'clicked'; }}
            return 'not_found';
        }})()
        """
        result = await self._cdp("Runtime.evaluate", {
            "expression": js_code,
            "returnByValue": True,
        })
        if result.get("error"):
            return result
        value = result.get("result", {}).get("result", {}).get("value", "unknown")
        return {"success": value == "clicked", "stdout": value}

    async def _type_text(self, selector: str, text: str) -> dict[str, Any]:
        """在输入框中输入文本。"""
        if not text:
            return {"error": "文本不能为空", "success": False}
        js_code = f"""
        (function() {{
            let el = document.querySelector('{selector}') || document.activeElement;
            if (!el) return 'no_element';
            el.focus();
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.contentEditable) {{
                el.value = '{text}';
                el.dispatchEvent(new Event('input', {{bubbles:true}}));
                return 'typed';
            }}
            return 'not_input';
        }})()
        """
        result = await self._cdp("Runtime.evaluate", {
            "expression": js_code,
            "returnByValue": True,
        })
        if result.get("error"):
            return result
        value = result.get("result", {}).get("result", {}).get("value", "unknown")
        return {"success": value in ("typed",), "stdout": value}

    async def _get_text(self) -> dict[str, Any]:
        """获取页面文本内容。"""
        result = await self._cdp("Runtime.evaluate", {
            "expression": "document.body ? document.body.innerText.substring(0, 5000) : 'no body'",
            "returnByValue": True,
        })
        if result.get("error"):
            return result
        value = result.get("result", {}).get("result", {}).get("value", "")
        return {"success": True, "text": value}

    async def _eval(self, js: str) -> dict[str, Any]:
        """执行 JavaScript 并返回结果。"""
        if not js:
            return {"error": "JS 不能为空", "success": False}
        result = await self._cdp("Runtime.evaluate", {
            "expression": js,
            "returnByValue": True,
        })
        if result.get("error"):
            return result
        value = result.get("result", {}).get("result", {}).get("value", "unknown")
        return {"success": True, "result": str(value)[:3000]}

    async def _fallback_navigate(self, url: str) -> dict[str, Any]:
        """CDP 导航失败 → 用 bash start 回退。"""
        proc = await asyncio.create_subprocess_shell(
            f'start {url}',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return {"success": True, "stdout": f"start {url}", "note": "CDP 不可用，使用系统默认浏览器"}

    async def close(self):
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
            risk=ToolRisk.MEDIUM,
            timeout_seconds=20,
            tags=["browser", "web", "cdp", "edge", "chrome"],
        )
