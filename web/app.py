"""DeepSoul Web UI — 基于 FastAPI 的 Web 控制面板。

提供:
- 对话界面
- 会话管理
- 系统状态监控
- 技能管理
- 记忆浏览
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from soul.engine.agent import Agent
from soul.config.manager import ConfigManager


def create_app(config_path: str | None = None) -> FastAPI:
    """创建 FastAPI 应用。"""

    app = FastAPI(
        title="DeepSoul Control Panel",
        description="DeepSoul Agent Web 控制面板",
        version="0.1.0",
    )

    # 初始化 Agent
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.load()
    agent: Agent | None = None
    agent_lock = asyncio.Lock()

    async def get_agent() -> Agent:
        nonlocal agent
        if agent is None:
            async with agent_lock:
                if agent is None:
                    agent = Agent(config=config)
                    await agent.initialize()
        return agent

    # ═══════════════════════════════════════
    # REST API
    # ═══════════════════════════════════════

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/api/status")
    async def api_status():
        try:
            a = await get_agent()
            return await a.get_status()
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/chat")
    async def api_chat(req: dict[str, Any]):
        """非流式对话 API。"""
        message = req.get("message", "")
        session_id = req.get("session_id", "")

        if not message:
            return JSONResponse({"error": "消息不能为空"}, status_code=400)

        a = await get_agent()
        response = await a.chat(message, session_id=session_id)
        return {"response": response, "session_id": session_id}

    @app.post("/api/sessions")
    async def api_create_session(req: dict[str, Any] | None = None):
        a = await get_agent()
        session = await a.sessions.create(
            session_key=req.get("key", "main") if req else "main",
        )
        return {
            "session_id": session.session_id,
            "session_key": session.session_key,
        }

    @app.get("/api/sessions")
    async def api_list_sessions():
        a = await get_agent()
        return await a.sessions.list_sessions()

    @app.get("/api/sessions/{session_id}/history")
    async def api_session_history(session_id: str, limit: int = 50):
        a = await get_agent()
        history = await a.sessions.get_history(session_id, last_n=limit)
        return [
            {
                "id": m.id,
                "role": m.role.value,
                "content": m.content[:1000],
                "timestamp": m.timestamp,
            }
            for m in history
        ]

    @app.delete("/api/sessions/{session_id}")
    async def api_close_session(session_id: str):
        a = await get_agent()
        await a.sessions.close(session_id)
        return {"ok": True}

    @app.get("/api/memory")
    async def api_memory_stats():
        a = await get_agent()
        return await a.memory.get_stats()

    @app.get("/api/skills")
    async def api_list_skills():
        a = await get_agent()
        return a.memory.procedural.list_skills()

    @app.post("/api/skills/search")
    async def api_search_skills(req: dict[str, Any]):
        a = await get_agent()
        query = req.get("query", "")
        skills = a.memory.procedural.match(query, top_k=5)
        return [
            {
                "name": s.meta.name,
                "description": s.meta.description,
                "version": s.meta.version,
                "fitness": s.meta.fitness_score,
            }
            for s in skills
        ]

    @app.get("/api/config")
    async def api_get_config():
        return cfg_mgr.config.model_dump()

    @app.post("/api/compact")
    async def api_compact(req: dict[str, Any] | None = None):
        a = await get_agent()
        session_id = req.get("session_id", "") if req else ""
        await a.compact(session_id)
        return {"ok": True}

    # ═══════════════════════════════════════
    # WebSocket 端点
    # ═══════════════════════════════════════

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        """WebSocket 流式对话。"""
        await websocket.accept()
        try:
            a = await get_agent()

            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")
                session_id = data.get("session_id", "")

                if not message:
                    await websocket.send_json({"error": "消息不能为空"})
                    continue

                # 流式发送回复
                async for chunk in a.chat_stream(message, session_id=session_id):
                    if chunk.content:
                        await websocket.send_json({
                            "type": "content",
                            "content": chunk.content,
                        })
                    if chunk.tool_call:
                        await websocket.send_json({
                            "type": "tool_call",
                            "tool": chunk.tool_call.name,
                        })
                    if chunk.finish_reason:
                        await websocket.send_json({
                            "type": "done",
                            "finish_reason": chunk.finish_reason,
                            "usage": chunk.usage,
                        })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.send_json({"error": str(e)})

    # ═══════════════════════════════════════
    # Web 页面
    # ═══════════════════════════════════════

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _get_index_html()

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page():
        return _get_chat_html()

    return app


def _get_index_html() -> str:
    """主页面 HTML。"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSoul Control Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        h1 { color: #58a6ff; margin-bottom: 2rem; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; }
        .card h2 { color: #f0f6fc; margin-bottom: 1rem; font-size: 1.1rem; }
        .card p { color: #8b949e; margin-bottom: 0.5rem; }
        .btn { display: inline-block; padding: 0.5rem 1rem; background: #238636; color: #fff; border: none; border-radius: 6px; cursor: pointer; text-decoration: none; margin-top: 1rem; }
        .btn:hover { background: #2ea043; }
        .status { font-size: 0.85rem; padding: 0.2rem 0.5rem; border-radius: 4px; }
        .status.ok { background: #23863622; color: #3fb950; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DeepSoul Control Panel</h1>
        <div class="cards">
            <div class="card">
                <h2>对话</h2>
                <p>启动与 DeepSoul Agent 的对话</p>
                <a href="/chat" class="btn">开始对话</a>
            </div>
            <div class="card">
                <h2>系统状态</h2>
                <p>查看 Agent 运行状态和统计信息</p>
                <div id="status">加载中...</div>
            </div>
            <div class="card">
                <h2>API 文档</h2>
                <p>查看完整的 REST API 和 WebSocket 文档</p>
                <a href="/docs" class="btn">API 文档</a>
            </div>
        </div>
    </div>
    <script>
        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status').innerHTML =
                    `<span class="status ok">运行中</span>
                     <p>模型: ${data.llm?.model || 'N/A'}</p>
                     <p>会话数: ${data.sessions?.active || 0}</p>
                     <p>工具数: ${data.tools?.total_tools || 0}</p>`;
            })
            .catch(() => {
                document.getElementById('status').innerHTML = '<span class="status" style="color:#f85149">未连接</span>';
            });
    </script>
</body>
</html>"""


def _get_chat_html() -> str:
    """对话页面 HTML。"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSoul Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0d1117; color: #c9d1d9; height: 100vh; display: flex; flex-direction: column; }
        header { background: #161b22; padding: 0.75rem 1.5rem; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
        header h1 { color: #58a6ff; font-size: 1rem; }
        #messages { flex: 1; overflow-y: auto; padding: 1rem; }
        .msg { margin-bottom: 1rem; padding: 0.5rem 1rem; border-radius: 6px; max-width: 80%; }
        .msg.user { background: #1f6feb22; border: 1px solid #1f6feb44; margin-left: auto; text-align: right; }
        .msg.assistant { background: #161b22; border: 1px solid #30363d; }
        .msg .role { font-size: 0.75rem; color: #8b949e; margin-bottom: 0.25rem; }
        .msg .content { white-space: pre-wrap; word-break: break-word; }
        #input-area { padding: 1rem; background: #161b22; border-top: 1px solid #30363d; display: flex; gap: 0.5rem; }
        #input { flex: 1; padding: 0.75rem; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; font-family: monospace; }
        #send { padding: 0.75rem 1.5rem; background: #238636; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-family: monospace; }
        #send:hover { background: #2ea043; }
        #send:disabled { opacity: 0.5; cursor: not-allowed; }
        .tool-call { color: #d2a8ff; font-size: 0.8rem; margin: 0.25rem 0; }
    </style>
</head>
<body>
    <header>
        <h1>DeepSoul Chat</h1>
        <span style="color:#8b949e;font-size:0.8rem;" id="session-id"></span>
    </header>
    <div id="messages"></div>
    <div id="input-area">
        <input type="text" id="input" placeholder="输入消息..." autofocus>
        <button id="send" onclick="send()">发送</button>
    </div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws/chat`);
        const messages = document.getElementById('messages');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        let currentAssistantMsg = null;
        let sessionId = '';

        ws.onopen = () => console.log('WebSocket 已连接');
        ws.onclose = () => console.log('WebSocket 已断开');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'content') {
                if (!currentAssistantMsg) {
                    currentAssistantMsg = addMsg('assistant', '');
                }
                currentAssistantMsg.querySelector('.content').textContent += data.content;
                messages.scrollTop = messages.scrollHeight;
            } else if (data.type === 'tool_call') {
                if (currentAssistantMsg) {
                    const tc = document.createElement('div');
                    tc.className = 'tool-call';
                    tc.textContent = '🔧 ' + data.tool;
                    currentAssistantMsg.appendChild(tc);
                }
            } else if (data.type === 'done') {
                currentAssistantMsg = null;
            }
        };

        function addMsg(role, content) {
            const div = document.createElement('div');
            div.className = 'msg ' + role;
            div.innerHTML = `<div class="role">${role === 'user' ? 'You' : 'DeepSoul'}</div><div class="content">${content}</div>`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            return div;
        }

        function send() {
            const text = input.value.trim();
            if (!text) return;

            addMsg('user', text);
            input.value = '';
            sendBtn.disabled = true;
            currentAssistantMsg = null;

            ws.send(JSON.stringify({ message: text, session_id: sessionId }));
            sendBtn.disabled = false;
        }

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') send();
        });

        fetch('/api/sessions', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                sessionId = data.session_id;
                document.getElementById('session-id').textContent = 'Session: ' + sessionId.slice(0,12);
            });
    </script>
</body>
</html>"""


def main():
    """启动 Web UI。"""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
