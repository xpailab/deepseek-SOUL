"""Gateway 服务器 — REST API + WebSocket + Web 聊天界面 + 并行多 Agent。

启动: soul gateway --port 18789
浏览器打开 http://localhost:18789
"""

import asyncio
import time
from typing import Any

from soul.engine.agent import Agent
from soul.engine.task_stages import TaskStagePlanner
from soul.gateway.router import ChannelMessage, MessageRouter
from soul.safety.auditor import Auditor
from soul.types import GatewayConfig


def _fmt_tool_result(tr) -> str:
    if tr.success:
        r = tr.result
        if isinstance(r, dict):
            parts = []
            if "exit_code" in r:
                parts.append(f"exit={r['exit_code']}")
            if r.get("stdout"):
                parts.append(r["stdout"][:500])
            if r.get("stderr"):
                parts.append("[stderr] " + r["stderr"][:300])
            return "\n".join(parts) if parts else str(r)[:500]
        return str(r)[:500]
    return tr.error or "执行失败"


class Gateway:
    """统一消息网关。"""

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig()
        self.router = MessageRouter()
        self.agent: Agent | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._stats: dict[str, Any] = {"messages_processed": 0, "errors": 0, "uptime_start": 0}
        self.auditor: Auditor | None = None
        self.connectors: dict[str, Any] = {}

    async def start(self, agent=None, host="", port=0):
        self.agent = agent
        self.auditor = agent.auditor if agent else None
        self._running = True
        self._stats["uptime_start"] = time.time()
        host = host or self.config.host
        port = port or self.config.port
        self._tasks.append(asyncio.create_task(self._serve(host, port)))
        print(f"\n  DeepSoul Gateway 已启动")
        display_host = "localhost" if host in ("0.0.0.0", "::", "") else host
        print(f"  ├─ Web 界面: http://{display_host}:{port}")
        print(f"  ├─ API 文档: http://{display_host}:{port}/docs")
        print(f"  └─ 健康检查: http://{display_host}:{port}/health")
        print(f"\n  按 Ctrl+C 停止\n")
        self.router.register_handler("cli", self._handle_cli_message)

    async def stop(self):
        self._running = False
        if self.auditor:
            self.auditor.flush()
        for connector in self.connectors.values():
            try:
                await connector.disconnect()
            except Exception:
                pass
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.agent:
            await self.agent.shutdown()

    async def handle_message(self, msg: ChannelMessage) -> dict[str, Any]:
        if not self.agent:
            return {"error": "Agent 未初始化"}
        self._stats["messages_processed"] += 1
        try:
            response = await self.agent.chat(msg.raw_text, session_id=msg.session_id)
            return {"reply": response, "session_id": msg.session_id}
        except Exception as e:
            self._stats["errors"] += 1
            return {"error": str(e)}

    async def _handle_cli_message(self, msg):
        return await self.handle_message(msg)

    async def register_connector(self, connector) -> None:
        """注册平台连接器并启动消息监听。"""
        self.connectors[connector.name] = connector
        if not connector.is_connected:
            await connector.connect()

        async def on_message(chat_id: str = "", text: str = "", user_name: str = "", **kwargs):
            msg = ChannelMessage(
                raw_text=text,
                channel=connector.name,
                channel_user_id=str(chat_id),
                channel_user_name=user_name,
            )
            await self.handle_message(msg)

        connector.on("message", on_message)

        if hasattr(connector, "listen") and callable(connector.listen):
            task = asyncio.create_task(connector.listen())
            self._tasks.append(task)

    async def _serve(self, host, port):
        try:
            import uvicorn
            from fastapi import FastAPI, Request, WebSocket
            from fastapi.responses import HTMLResponse, JSONResponse

            app = FastAPI()

            @app.get("/", response_class=HTMLResponse)
            async def index():
                return CHAT_PAGE

            @app.get("/health")
            async def health():
                return {"status": "ok", "uptime": round(time.time() - self._stats["uptime_start"], 1)}

            @app.post("/api/chat")
            async def api_chat(req: dict[str, Any]):
                msg = req.get("message", "")
                sid = req.get("session_id", "")
                if not msg:
                    return JSONResponse({"error": "消息为空"}, 400)
                a = self.agent
                if not a:
                    return JSONResponse({"error": "Agent 未就绪"}, 503)
                if self.auditor:
                    self.auditor.record("api_access", {
                        "endpoint": "/api/chat",
                        "session_id": sid,
                        "message_len": len(msg),
                    })
                reply = await a.chat(msg, session_id=sid)
                return {"reply": reply, "session_id": sid}

            @app.post("/api/sessions")
            async def api_create_session(req: dict[str, Any] | None = None):
                a = self.agent
                if not a:
                    return JSONResponse({"error": "Agent 未就绪"}, 503)
                s = await a.sessions.create(session_key=(req or {}).get("key", "main"))
                return {"session_id": s.session_id}

            @app.get("/api/status")
            async def api_status():
                a = self.agent
                return await a.get_status() if a else {"error": "Agent 未就绪"}

            @app.get("/api/sessions")
            async def api_list_sessions():
                a = self.agent
                return await a.sessions.list_sessions() if a else []

            @app.get("/api/audit")
            async def api_audit(event_type: str = "", severity: str = "", limit: int = 50):
                if not self.auditor:
                    return JSONResponse({"error": "审计未启用"}, 503)
                events = self.auditor.query(
                    event_type=event_type, severity=severity, limit=limit
                )
                return {"events": events, "count": len(events)}

            @app.get("/api/audit/report")
            async def api_audit_report():
                if not self.auditor:
                    return JSONResponse({"error": "审计未启用"}, 503)
                return self.auditor.get_security_report()

            @app.post("/webhook/{platform}")
            async def webhook_receiver(platform: str, request: Request):
                """接收平台 Webhook 回调，路由到 Agent 处理。"""
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse({"error": "无效的 JSON 请求体"}, 400)

                # URL 验证（飞书/钉钉）
                if platform == "feishu" and body.get("type") == "url_verification":
                    return {"challenge": body.get("challenge", "")}
                if platform == "dingtalk" and "test" in body:
                    return {"status": "ok"}

                if not self.agent:
                    return JSONResponse({"error": "Agent 未就绪"}, 503)

                # 解析平台消息
                text = ""
                sender_id = ""
                if platform == "qq":
                    text = body.get("content", "")
                    sender_id = body.get("author", {}).get("id", "")
                elif platform == "wechat":
                    text = body.get("Content", "")
                    sender_id = body.get("FromUserName", "")
                elif platform == "dingtalk":
                    text = body.get("text", {}).get("content", "")
                    sender_id = body.get("senderStaffId", "")
                elif platform == "feishu":
                    event = body.get("event", {})
                    text = event.get("text", "")
                    sender_id = event.get("sender", {}).get("sender_id", "")
                elif platform == "telegram":
                    msg = body.get("message", {})
                    text = msg.get("text", "")
                    sender_id = str(msg.get("chat", {}).get("id", ""))

                if not text:
                    return {"status": "ignored", "reason": "empty message"}

                # 通过 Agent 处理消息
                sid = f"{platform}:{sender_id}"
                reply = await self.agent.chat(text, session_id=sid)

                # 通过连接器回复
                connector = self.connectors.get(platform)
                if connector and connector.is_connected:
                    await connector.send_message(reply, chat_id=sender_id)

                return {"status": "ok", "reply": reply[:200], "session_id": sid}

            @app.websocket("/ws/chat")
            async def ws_chat(ws: WebSocket):
                await ws.accept()
                try:
                    while self._running:
                        data = await ws.receive_json()
                        txt = data.get("message", "")
                        sid = data.get("session_id", "")
                        if not txt or not self.agent:
                            continue
                        if self.auditor:
                            self.auditor.record("api_access", {
                                "endpoint": "/ws/chat",
                                "session_id": sid,
                                "message_len": len(txt),
                            })
                        try:
                            from soul.engine.parallel import ParallelAgent, _llm_classify
                            base_cfg = self.agent.config

                            # LLM 判断复杂度
                            adapter = self.agent.llm.get(base_cfg.llm)
                            difficulty = await _llm_classify(txt, adapter)

                            if difficulty <= 1:
                                # 简单对话 → 直接回复
                                try:
                                    async for chunk in self.agent.chat_stream(txt, session_id=sid):
                                        d = {}
                                        if chunk.content: d["c"] = chunk.content
                                        if chunk.tool_call: d["t"] = chunk.tool_call.name
                                        if chunk.tool_result:
                                            d["r"] = {"ok": chunk.tool_result.success,
                                                       "text": _fmt_tool_result(chunk.tool_result)}
                                        if chunk.finish_reason: d["f"] = chunk.finish_reason
                                        if d: await ws.send_json(d)
                                except Exception as e:
                                    await ws.send_json({"stream_id": "_meta", "type": "error", "content": str(e), "f": "error"})
                                # 确保流结束标记
                                await ws.send_json({"f": "stop"})
                            elif difficulty >= 4:
                                # 复杂项目 → 分阶段执行
                                try:
                                    # 先进行任务规划
                                    planner = TaskStagePlanner(self.agent)
                                    plan = await planner.plan(txt)

                                    if planner.should_use_stages(txt, plan):
                                        # 发送阶段计划给用户
                                        await ws.send_json({
                                            "stream_id": "_meta",
                                            "type": "stage_plan",
                                            "plan": plan.to_dict(),
                                            "message": f"任务已拆分为 {len(plan.stages)} 个阶段，预计需要 {plan.total_estimated_tools} 次工具调用。"
                                        })

                                        # 逐阶段执行
                                        previous_results = []
                                        while not plan.is_complete():
                                            stage = plan.get_current_stage()
                                            if not stage:
                                                break

                                            # 构建阶段提示
                                            stage_prompt = build_stage_prompt(stage, plan, previous_results)

                                            # 执行当前阶段
                                            await ws.send_json({
                                                "stream_id": "_meta",
                                                "type": "stage_start",
                                                "stage": stage.to_dict(),
                                                "progress": plan.get_progress_summary(),
                                            })

                                            stage_content = ""
                                            async for chunk in self.agent.chat_stream(
                                                txt,  # 原始任务
                                                session_id=sid,
                                                system_prompt=stage_prompt,
                                            ):
                                                d = {"stage_id": stage.id}
                                                if chunk.content:
                                                    d["c"] = chunk.content
                                                    stage_content += chunk.content
                                                if chunk.tool_call: d["t"] = chunk.tool_call.name
                                                if chunk.tool_result:
                                                    d["r"] = {"ok": chunk.tool_result.success,
                                                               "text": _fmt_tool_result(chunk.tool_result)}
                                                if chunk.finish_reason: d["f"] = chunk.finish_reason
                                                if len(d) > 1:
                                                    await ws.send_json(d)

                                            # 解析阶段完成结果
                                            summary, artifacts = parse_stage_completion(stage_content)
                                            plan.complete_current_stage(summary, artifacts)
                                            previous_results.append(summary)

                                            # 发送阶段完成消息
                                            await ws.send_json({
                                                "stream_id": "_meta",
                                                "type": "stage_complete",
                                                "stage": stage.to_dict(),
                                                "summary": summary,
                                                "artifacts": artifacts,
                                                "progress": plan.get_progress_summary(),
                                            })

                                            # 如果不是最后一个阶段，等待用户确认
                                            if not plan.is_complete():
                                                next_stage = plan.get_next_stage()
                                                await ws.send_json({
                                                    "stream_id": "_meta",
                                                    "type": "await_confirm",
                                                    "message": f"阶段 '{stage.name}' 完成。",
                                                    "next_stage": next_stage.to_dict() if next_stage else None,
                                                    "progress": plan.get_progress_summary(),
                                                })
                                                # 这里需要等待前端发送确认消息
                                                # 暂时直接继续（后续改进）
                                                await asyncio.sleep(0.5)

                                        # 所有阶段完成
                                        await ws.send_json({
                                            "stream_id": "_meta",
                                            "type": "all_stages_complete",
                                            "plan": plan.to_dict(),
                                            "message": "所有阶段已完成！",
                                        })
                                    else:
                                        # 不需要分阶段，使用普通模式
                                        async for chunk in self.agent.chat_stream(txt, session_id=sid):
                                            d = {}
                                            if chunk.content: d["c"] = chunk.content
                                            if chunk.tool_call: d["t"] = chunk.tool_call.name
                                            if chunk.tool_result:
                                                d["r"] = {"ok": chunk.tool_result.success,
                                                           "text": _fmt_tool_result(chunk.tool_result)}
                                            if chunk.finish_reason: d["f"] = chunk.finish_reason
                                            if d: await ws.send_json(d)

                                except Exception as e:
                                    await ws.send_json({"stream_id": "_meta", "type": "error", "content": str(e), "f": "error"})
                                finally:
                                    await ws.send_json({"f": "stop"})
                            else:
                                # 中等复杂度 → 并行Agent
                                pa = None
                                try:
                                    async def make_agent():
                                        a = Agent(config=base_cfg)
                                        await a.initialize()
                                        return a
                                    pa = ParallelAgent(make_agent)
                                    async for evt in pa.execute(txt, sid, agent_count=difficulty):
                                        await ws.send_json(evt)
                                except Exception as e:
                                    await ws.send_json({"stream_id": "_meta", "type": "error", "content": str(e), "f": "error"})
                                finally:
                                    # 确保流结束标记
                                    await ws.send_json({"stream_id": "_meta", "type": "finished", "f": "stop"})
                        except Exception as e:
                            await ws.send_json({"stream_id": "_meta", "type": "error", "content": str(e), "f": "error"})
                except Exception:
                    pass

            config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off")
            await uvicorn.Server(config).serve()
        except ImportError:
            await self._serve_fallback(host, port)

    async def _serve_fallback(self, host, port):
        async def handle(reader, writer):
            try:
                data = await reader.read(4096)
                if data:
                    text = data.decode("utf-8", errors="replace").strip()
                    msg = ChannelMessage(raw_text=text, channel="tcp",
                                         channel_user_id=f"{writer.get_extra_info('peername')}")
                    result = await self.handle_message(msg)
                    writer.write(str(result).encode("utf-8") + b"\n")
                    await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
        server = await asyncio.start_server(handle, host, port)
        async with server:
            await server.serve_forever()


CHAT_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeepSoul</title>
<script>
  // marked.js 多 CDN 容错加载
  (function loadMarked(urls,i){
    var s=document.createElement('script'); s.src=urls[i];
    s.onload=function(){ console.log('marked loaded from '+urls[i]); };
    s.onerror=function(){
      if(i+1<urls.length){ loadMarked(urls,i+1); }
      else{ console.warn('marked.js 所有CDN加载失败, 降级为纯文本'); window._markedFail=true; }
    };
    document.head.appendChild(s);
  })([
    'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
    'https://unpkg.com/marked/marked.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js',
  ],0);
</script>
<style>
  :root {
    --bg: #f8f9fb; --surface: #fff; --border: #e5e7eb;
    --text: #1f2937; --text-dim: #9ca3af; --text-secondary: #6b7280;
    --accent: #4f46e5; --accent-light: #eef2ff;
    --user-bg: #f0f2ff; --user-border: #dde0ff;
    --tool-bg: #f9fafb; --tool-border: #e5e7eb;
    --danger: #ef4444; --success: #10b981;
    --radius: 12px; --radius-sm: 8px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',sans-serif; background:var(--bg); color:var(--text); height:100vh; display:flex; font-size:15px; line-height:1.6; }
  .sidebar { width:260px; background:var(--surface); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:16px; flex-shrink:0; }
  .sidebar-logo { font-size:1.15rem; font-weight:700; color:var(--accent); padding:8px 12px; margin-bottom:20px; display:flex; align-items:center; gap:8px; }
  .sidebar-logo svg { width:24px; height:24px; }
  .btn-new { width:100%; padding:10px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--surface); color:var(--text); cursor:pointer; font-size:.85rem; transition:.15s; text-align:left; margin-bottom:12px; }
  .btn-new:hover { background:var(--accent-light); border-color:var(--accent); }
  .sidebar-info { font-size:.72rem; color:var(--text-dim); margin-top:auto; padding:12px; border-top:1px solid var(--border); }
  .sidebar-info span { display:block; margin:2px 0; }
  .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; }
  .status-dot.on { background:var(--success); }
  .main { flex:1; display:flex; flex-direction:column; min-width:0; background:var(--bg); }
  .header { padding:12px 24px; background:var(--surface); border-bottom:1px solid var(--border); font-size:.85rem; color:var(--text-secondary); }
  .header strong { color:var(--text); font-weight:600; }
  .messages { flex:1; overflow-y:auto; padding:24px 0; }
  .messages-inner { max-width:800px; margin:0 auto; padding:0 24px; display:flex; flex-direction:column; gap:4px; }
  .messages::-webkit-scrollbar { width:5px; }
  .messages::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
  .msg { padding:12px 20px; animation:fadeIn .3s ease; display:flex; }
  .msg:empty { display:none; }
  .msg:has(.msg-text:empty) { display:none; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  .msg.user { background:var(--user-bg); justify-content:flex-end; }
  .msg.assistant { background:var(--surface); justify-content:flex-start; }
  .msg-body { max-width:75%; min-width:120px; white-space:normal; word-break:normal; }
  .msg-body p { margin:.4em 0; }
  .msg-body pre { background:#1e1e2e; color:#cdd6f4; border-radius:var(--radius-sm); padding:14px 18px; overflow-x:auto; margin:8px 0; font-size:.82rem; line-height:1.55; }
  .msg-body code { font-family:'SF Mono','Cascadia Code',Consolas,monospace; font-size:.85em; }
  .msg-body :not(pre)>code { background:#f3f4f6; padding:2px 6px; border-radius:3px; color:#e01b7b; }
  .msg-body pre code { background:none; padding:0; color:inherit; }

  /* Agent Cards */
  .agents-row {
    display: flex; flex-direction: column; gap: 6px; margin: 8px 0; width: 100%;
  }
  .agent-card {
    width: 100%; max-height: 320px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-sm); overflow: hidden;
    font-size: .73rem; transition: all .3s; display: flex; flex-direction: column;
  }
  .agent-card.winner {
    border-color: var(--success);
    box-shadow: 0 0 14px rgba(16,185,129,.2);
  }
  .agent-card.loser { opacity: .4; }
  .agent-card .card-head {
    padding: 7px 10px; background: var(--accent-light);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none;
    flex-shrink: 0;
  }
  .agent-card .card-head .c-name { font-weight: 600; font-size: .72rem; color: var(--accent); flex:1; }
  .agent-card .card-head .c-status {
    font-size: .62rem; padding: 2px 6px; border-radius: 3px;
    background: #f3f4f6; color: var(--text-dim);
  }
  .agent-card.winner .card-head .c-status { background: #d1fae5; color: #065f46; }
  .agent-card.loser .card-head .c-status { background: #fee2e2; color: #991b1b; }
  .agent-card .card-head .arrow { font-size: .55rem; transition: transform .15s; color: var(--text-dim); }
  .agent-card .card-head .arrow.open { transform: rotate(90deg); }
  .agent-card .card-body {
    padding: 8px 10px; flex: 1; overflow-y: auto;
    line-height: 1.5; white-space: normal; word-break: normal; min-width: 0;
  }
  .agent-card .card-body .c-text { white-space: normal; word-break: normal; }
  .agent-card .card-body.collapsed { display: none; }
  .agent-card .card-body .c-text {
    font-size: .75rem; color: var(--text); font-weight: 500;
    margin-bottom: 4px;
  }
  .agent-card .card-body .c-tool {
    font-size: .62rem; color: var(--text-dim); padding: 2px 0;
    border-top: 1px dotted var(--border); margin-top: 4px;
  }
  .agent-card .card-body .c-tool.ok { color: var(--success); }
  .agent-card .card-body .c-tool.err { color: var(--danger); }

  /* Markdown 渲染 */
  .msg-text h1,.msg-text h2,.msg-text h3 { margin: .8em 0 .3em; line-height:1.3; }
  .msg-text h1 { font-size:1.3rem; }
  .msg-text h2 { font-size:1.15rem; }
  .msg-text h3 { font-size:1.0rem; color:var(--text-secondary); }
  .msg-text ul,.msg-text ol { margin:.4em 0; padding-left:1.5em; }
  .msg-text li { margin:.15em 0; }
  .msg-text blockquote {
    border-left:3px solid var(--accent); margin:.5em 0; padding:.2em .8em;
    background:var(--accent-light); border-radius:0 var(--radius-sm) var(--radius-sm) 0;
    color:var(--text-secondary); font-size:.92em;
  }
  .msg-text blockquote p { margin:.3em 0; }
  .msg-text hr { border:none; border-top:1px solid var(--border); margin:1em 0; }
  .msg-text table { border-collapse:collapse; width:100%; margin:.5em 0; font-size:.85em; }
  .msg-text th,.msg-text td { border:1px solid var(--border); padding:6px 10px; text-align:left; }
  .msg-text th { background:var(--accent-light); font-weight:600; }
  .msg-text tr:nth-child(even) { background:#fafbfc; }
  .msg-text a { color:var(--accent); text-decoration:none; }
  .msg-text a:hover { text-decoration:underline; }
  .msg-text img { max-width:100%; border-radius:var(--radius-sm); }
  .msg-text strong { font-weight:600; }

  .welcome { text-align:center; padding:60px 20px; color:var(--text-dim); }
  .welcome-icon { font-size:3rem; margin-bottom:16px; }
  .welcome h2 { color:var(--text); font-size:1.4rem; margin-bottom:8px; font-weight:600; }
  .input-area { background:var(--surface); padding:16px 0 20px; }
  .input-inner { max-width:800px; margin:0 auto; padding:0 24px; }
  .input-wrap { display:flex; gap:10px; align-items:flex-end; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius); padding:8px 8px 8px 18px; transition:border-color .15s,box-shadow .15s; }
  .input-wrap:focus-within { border-color:var(--accent); box-shadow:0 0 0 3px rgba(79,70,229,.1); background:var(--surface); }
  .input-wrap textarea { flex:1; background:none; border:none; color:var(--text); font-size:.9rem; resize:none; outline:none; padding:6px 0; font-family:inherit; line-height:1.5; max-height:140px; }
  .input-wrap textarea::placeholder { color:var(--text-dim); }
  .input-wrap button { width:38px; height:38px; border:none; border-radius:10px; background:var(--accent); color:#fff; cursor:pointer; flex-shrink:0; transition:.15s; display:flex; align-items:center; justify-content:center; }
  .input-wrap button:hover { opacity:.88; }
  .input-wrap button:disabled { opacity:.35; cursor:not-allowed; }
  .input-wrap button svg { width:16px; height:16px; }
  .input-hint { font-size:.7rem; color:var(--text-dim); margin-top:8px; text-align:center; }
  @media (max-width:720px) { .sidebar{display:none} .messages-inner,.input-inner{padding:0 12px} }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
    DeepSoul
  </div>
  <button class="btn-new" onclick="newChat()">+ 新对话</button>
  <div id="sessionDisplay" class="sidebar-info">
    <span><span class="status-dot on"></span> 在线</span>
    <span id="modelName" style="color:var(--text-secondary)">-</span>
    <span id="msgCount" style="color:var(--text-dim)">就绪</span>
  </div>
</div>
<div class="main">
  <div class="header"><strong>DeepSoul</strong> Agent</div>
  <div class="messages" id="messages">
    <div class="messages-inner" id="msgList">
      <div class="welcome" id="welcome">
        <div class="welcome-icon">&#9670;</div>
        <h2>有什么我可以帮你的？</h2>
        <p style="color:var(--text-dim)">我可以操控电脑、写代码、查资料、管理项目</p>
      </div>
    </div>
  </div>
  <div class="input-area">
    <div class="input-inner">
      <div class="input-wrap">
        <textarea id="input" rows="1" placeholder="输入消息，Enter 发送，Shift+Enter 换行" onkeydown="onKey(event)"></textarea>
        <button id="sendBtn" onclick="send()" title="发送">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
      <div class="input-hint">Enter 发送 · Shift+Enter 换行 · 并行 Agent 自动分配</div>
    </div>
  </div>
</div>
<script>
  // 安全的 markdown 渲染（CDN 失败时降级为纯文本转义）
  function renderMD(text) {
    if (typeof marked === 'undefined' || window._markedFail) {
      var d = document.createElement('div'); d.textContent = text; return d.innerHTML;
    }
    if (!window._markedInit) {
      marked.setOptions({gfm:true, breaks:false});
      window._markedInit = true;
    }
    var clean = text;
    // 1. 修复 CJK 字符间多余空格
    clean = clean.replace(/([\u4e00-\u9fff\u3400-\u4dbf])\\s+([\u4e00-\u9fff\u3400-\u4dbf])/g, '$1$2');
    // 2. 保护双下划线变量名 (如 __init__, __name__) 不被 marked 吃掉
    clean = clean.replace(/__([a-zA-Z0-9]+)__/g, '\\\\_\\\\_$1\\\\_\\\\_');
    // 3. 保护单词内的下划线 (如 py_project, file_name.py)
    clean = clean.replace(/([a-zA-Z0-9])_([a-zA-Z0-9])/g, '$1\\\\_$2');
    // 4. 保护 CJK 字符前后可能被误解析的下划线
    clean = clean.replace(/([\u4e00-\u9fff])_(\\w)/g, '$1\\\\_$2');
    clean = clean.replace(/(\\w)_([\u4e00-\u9fff])/g, '$1\\\\_$2');
    return marked.parse(clean);
  }

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  let ws = null, sessionId = '', wsReady = false, reconnectTimer = null;
  let agentCards = {}, agentsRow = null;

  function toast(t) {
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;bottom:80px;left:50%;transform:translateX(-50%);background:#ef4444;color:#fff;padding:8px 18px;border-radius:8px;font-size:.82rem;z-index:999';
    el.textContent = t; document.body.appendChild(el);
    setTimeout(()=>{el.style.opacity='0';el.style.transition='opacity .5s'},2000);
    setTimeout(()=>el.remove(),2500);
  }

  async function init() {
    try{ const r = await fetch('/api/sessions',{method:'POST'}); const d = await r.json(); sessionId = d.session_id; document.getElementById('sessionDisplay').innerHTML = '<span><span class=\"status-dot on\"></span> 在线</span><span id=\"modelName\" style=\"color:var(--text-secondary)\">-</span><span id=\"msgCount\" style=\"color:var(--text-dim)\">就绪</span>'; }catch(e){}
    connect();
    try{ const r = await fetch('/api/status'); const d = await r.json(); if(d.llm) document.getElementById('modelName').textContent = d.llm.model || '-'; }catch(e){}
  }

  function connect() {
    if(ws && (ws.readyState===WebSocket.CONNECTING || ws.readyState===WebSocket.OPEN)) return;
    ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);
    ws.onopen = () => { wsReady = true; document.getElementById('sendBtn').disabled = false; document.getElementById('input').focus(); };
    ws.onmessage = handleMessage;
    ws.onerror = () => { wsReady = false; };
    ws.onclose = () => { wsReady = false; if(reconnectTimer) clearTimeout(reconnectTimer); reconnectTimer = setTimeout(connect,2000); };
  }

  let simpleMsg = null;  // 简单对话的当前消息元素

  function handleMessage(e) {
    let d; try{ d = JSON.parse(e.data); }catch(_){ return; }
    const sid = d.stream_id || '';

    // === 简单对话模式 (无 stream_id，旧格式) ===
    if(!sid){
      if(d.c){
        if(!simpleMsg){
          document.getElementById('welcome')?.remove();
          simpleMsg = document.createElement('div'); simpleMsg.className = 'msg assistant';
          simpleMsg.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\"></div></div>';
          document.getElementById('msgList').appendChild(simpleMsg);
        }
        simpleMsg.querySelector('.msg-text').innerHTML = renderMD(simpleMsg.querySelector('.msg-text').textContent + d.c);
        scrollDown();
      }
      if(d.t){
        if(!simpleMsg){
          document.getElementById('welcome')?.remove();
          simpleMsg = document.createElement('div'); simpleMsg.className = 'msg assistant';
          simpleMsg.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\"></div></div>';
          document.getElementById('msgList').appendChild(simpleMsg);
        }
        const tl = document.createElement('div');
        tl.style.cssText = 'font-size:.7rem;color:var(--text-dim);margin:2px 0';
        tl.textContent = '...' + d.t; simpleMsg.querySelector('.msg-body').appendChild(tl);
        scrollDown();
      }
      if(d.r){
        const rl = document.createElement('div');
        rl.style.cssText = 'font-size:.68rem;margin:1px 0;color:' + (d.r.ok?'var(--success)':'var(--danger)');
        rl.textContent = (d.r.ok?'OK ':'FAIL ') + ': ' + (d.r.text||'').slice(0,80);
        if(simpleMsg) simpleMsg.querySelector('.msg-body').appendChild(rl);
        scrollDown();
      }
      if(d.f){
        simpleMsg = null;
        document.getElementById('sendBtn').disabled = false;
        document.getElementById('input').focus();
        document.getElementById('msgCount').textContent = document.querySelectorAll('.msg').length + ' 条消息';
      }
      return;
    }

    // === 并行模式 (有 stream_id) ===

    // Meta: 并行启动
    if(sid === '_meta' && d.type === 'start'){
      document.getElementById('welcome')?.remove();
      const count = d.count || 1;
      // 单 Agent → 不显示卡片，用正常消息流
      if(count <= 1){
        const div = document.createElement('div'); div.className = 'msg assistant';
        div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\"></div></div>';
        div.id = 'singleAgentMsg';
        document.getElementById('msgList').appendChild(div);
        agentCards['0'] = {msgDiv: div, body: div.querySelector('.msg-text')};
        scrollDown(); return;
      }
      // 多 Agent → 横向卡片布局（默认折叠）
      agentsRow = document.createElement('div'); agentsRow.className = 'agents-row';
      document.getElementById('msgList').appendChild(agentsRow);
      (d.approaches||[]).forEach((name,i) => {
        const card = document.createElement('div'); card.className = 'agent-card';
        card.innerHTML = '<div class=\"card-head\"><span class=\"arrow\">&#9654;</span><span class=\"c-name\">' + name + '</span><span class=\"c-status\">等待</span></div><div class=\"card-body collapsed\"></div>';
        card.querySelector('.card-head').onclick = function(){
          const body = card.querySelector('.card-body');
          const arrow = card.querySelector('.arrow');
          body.classList.toggle('collapsed');
          arrow.classList.toggle('open');
        };
        agentsRow.appendChild(card);
        agentCards[i] = card;
      });
      scrollDown(); return;
    }

    // Meta: 完成
    if(sid === '_meta' && (d.type === 'finished' || d.type === 'partial')){
      const w = d.winner !== undefined ? String(d.winner) : null;
      const hasCards = agentsRow !== null;

      if(hasCards){
        Object.keys(agentCards).forEach(k => {
          const card = agentCards[k];
          const st = card.querySelector('.c-status');
          if(k === w){ card.classList.add('winner'); st.textContent = '采用'; st.style.color = '#065f46'; }
          else { card.classList.add('loser'); st.textContent = '未用'; st.style.color = '#991b1b'; }
        });
        // 追加最终结果（含任务报告）
        const div = document.createElement('div'); div.className = 'msg assistant';
        const winnerLabel = d.winner_name ? '【采用: ' + d.winner_name + '】' : '';
        div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + winnerLabel + (d.content ? renderMD(d.content) : '完成') + '</div></div>';
        document.getElementById('msgList').appendChild(div);
      }
      document.getElementById('sendBtn').disabled = false;
      document.getElementById('input').focus();
      document.getElementById('msgCount').textContent = document.querySelectorAll('.msg').length + ' 条消息';
      agentCards = {}; agentsRow = null;
      scrollDown(); return;
    }

    // Meta: 阶段计划
    if(sid === '_meta' && d.type === 'stage_plan'){
      const div = document.createElement('div'); div.className = 'msg assistant stage-plan';
      let stagesHtml = '<div style=\"background:var(--accent-light);border:1px solid var(--accent);border-radius:8px;padding:12px;margin:8px 0;\">';
      stagesHtml += '<div style=\"font-weight:600;color:var(--accent);margin-bottom:8px;\">📋 任务执行计划</div>';
      stagesHtml += '<div style=\"font-size:.85rem;color:var(--text-secondary);margin-bottom:8px;\">' + (d.message||'') + '</div>';
      if(d.plan && d.plan.stages){
        stagesHtml += '<div style=\"display:flex;flex-direction:column;gap:4px;\">';
        d.plan.stages.forEach((stage, idx) => {
          stagesHtml += '<div style=\"display:flex;align-items:center;gap:8px;padding:6px 8px;background:var(--bg);border-radius:4px;\">';
          stagesHtml += '<span style=\"background:var(--accent);color:white;padding:2px 6px;border-radius:3px;font-size:.7rem;\">阶段' + (idx+1) + '</span>';
          stagesHtml += '<span style=\"font-weight:500;\">' + stage.name + '</span>';
          stagesHtml += '<span style=\"margin-left:auto;font-size:.75rem;color:var(--text-dim);\">预计' + stage.estimated_tools + '步</span>';
          stagesHtml += '</div>';
        });
        stagesHtml += '</div>';
      }
      stagesHtml += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + stagesHtml + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      scrollDown(); return;
    }

    // Meta: 阶段开始
    if(sid === '_meta' && d.type === 'stage_start'){
      const div = document.createElement('div'); div.className = 'msg assistant stage-start';
      let html = '<div style=\"background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;\">';
      html += '<span style=\"font-size:1.2rem;\">▶️</span>';
      html += '<span style=\"font-weight:600;color:#166534;\">开始执行: ' + (d.stage?.name||'当前阶段') + '</span>';
      html += '<span style=\"margin-left:auto;font-size:.75rem;color:#22c55e;\">' + (d.progress||'') + '</span>';
      html += '</div>';
      if(d.stage?.description){
        html += '<div style=\"margin-top:6px;font-size:.85rem;color:var(--text-secondary);padding-left:28px;\">' + d.stage.description + '</div>';
      }
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      scrollDown(); return;
    }

    // Meta: 阶段完成
    if(sid === '_meta' && d.type === 'stage_complete'){
      const div = document.createElement('div'); div.className = 'msg assistant stage-complete';
      let html = '<div style=\"background:#f0f9ff;border:1px solid #7dd3fc;border-radius:8px;padding:10px 12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;\">';
      html += '<span style=\"font-size:1.2rem;\">✅</span>';
      html += '<span style=\"font-weight:600;color:#0369a1;\">阶段完成: ' + (d.stage?.name||'') + '</span>';
      html += '</div>';
      if(d.summary){
        html += '<div style=\"margin-top:6px;font-size:.85rem;color:var(--text);padding-left:28px;\">' + d.summary + '</div>';
      }
      if(d.artifacts && d.artifacts.length > 0){
        html += '<div style=\"margin-top:6px;font-size:.8rem;color:var(--text-dim);padding-left:28px;\">';
        html += '📁 交付物: ' + d.artifacts.join(', ');
        html += '</div>';
      }
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      scrollDown(); return;
    }

    // Meta: 等待确认
    if(sid === '_meta' && d.type === 'await_confirm'){
      const div = document.createElement('div'); div.className = 'msg assistant await-confirm';
      let html = '<div style=\"background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;padding:12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;margin-bottom:8px;\">';
      html += '<span style=\"font-size:1.2rem;\">⏸️</span>';
      html += '<span style=\"font-weight:600;color:#92400e;\">' + (d.message||'阶段完成') + '</span>';
      html += '</div>';
      if(d.next_stage){
        html += '<div style=\"font-size:.85rem;color:var(--text-secondary);padding-left:28px;margin-bottom:8px;\">';
        html += '下一阶段: <strong>' + d.next_stage.name + '</strong> - ' + d.next_stage.description;
        html += '</div>';
      }
      html += '<div style=\"padding-left:28px;\">';
      html += '<button onclick=\"continueStage()\" style=\"background:#f59e0b;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:.85rem;\">继续下一阶段 →</button>';
      html += '<span style=\"margin-left:12px;font-size:.8rem;color:var(--text-dim);\">' + (d.progress||'') + '</span>';
      html += '</div>';
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      div.id = 'awaitConfirmMsg';
      document.getElementById('msgList').appendChild(div);
      document.getElementById('sendBtn').disabled = true;
      scrollDown(); return;
    }

    // Meta: 所有阶段完成
    if(sid === '_meta' && d.type === 'all_stages_complete'){
      const div = document.createElement('div'); div.className = 'msg assistant all-complete';
      let html = '<div style=\"background:#d1fae5;border:1px solid #34d399;border-radius:8px;padding:12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;\">';
      html += '<span style=\"font-size:1.5rem;\">🎉</span>';
      html += '<span style=\"font-weight:600;color:#065f46;font-size:1.1rem;\">所有阶段已完成！</span>';
      html += '</div>';
      if(d.plan && d.plan.stages){
        html += '<div style=\"margin-top:8px;font-size:.85rem;color:var(--text);padding-left:36px;\">';
        html += '共完成 ' + d.plan.stages.length + ' 个阶段';
        html += '</div>';
      }
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      document.getElementById('sendBtn').disabled = false;
      scrollDown(); return;
    }

    // Meta: 错误
    if(sid === '_meta' && d.type === 'error'){
      const div = document.createElement('div'); div.className = 'msg assistant';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\" style=\"color:var(--danger)\">' + (d.content ? renderMD(d.content) : '错误') + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      document.getElementById('sendBtn').disabled = false;
      agentCards = {}; agentsRow = null;
      scrollDown(); return;
    }

    // Agent 事件
    if(sid && sid !== '_meta' && agentCards[sid]){
      const entry = agentCards[sid];
      // 单 Agent 消息模式
      if(entry.msgDiv){
        if(d.type === 'content'){ entry.body.innerHTML = renderMD((entry.body.textContent||'') + d.content); }
        else if(d.type === 'tool'){
          const tl = document.createElement('div');
          tl.style.cssText = 'font-size:.7rem;color:var(--text-dim);margin:2px 0';
          tl.textContent = '...' + d.tool; entry.body.appendChild(tl);
        }
        else if(d.type === 'result'){
          const rl = document.createElement('div');
          rl.style.cssText = 'font-size:.68rem;margin:1px 0;color:' + (d.success?'var(--success)':'var(--danger)');
          rl.textContent = (d.success?'OK ':'FAIL ') + (d.tool||'') + ': ' + (d.text||'').slice(0,60);
          entry.body.appendChild(rl);
        }
      }
      // 多 Agent 卡片模式
      else {
        const body = entry.querySelector('.card-body');
        const status = entry.querySelector('.c-status');
        if(d.type === 'agent_start'){ status.textContent = '执行'; status.style.color = '#3b82f6'; }
        else if(d.type === 'content'){
          if(body.classList.contains('collapsed')){
            body.classList.remove('collapsed');
            entry.querySelector('.arrow').classList.add('open');
          }
          // 追加到同一个文本行，不创建新 div
          let last = body.querySelector('.c-text:last-child');
          if(!last){ last = document.createElement('div'); last.className = 'c-text'; body.appendChild(last); }
          last.innerHTML = renderMD((last.textContent||'') + d.content);
          body.scrollTop = body.scrollHeight;
        }
        else if(d.type === 'tool'){
          status.textContent = '执行'; status.style.color = '#3b82f6';
          const tl = document.createElement('div'); tl.className = 'c-tool';
          tl.textContent = '... ' + d.tool; body.appendChild(tl);
          body.scrollTop = body.scrollHeight;
        }
        else if(d.type === 'result'){
          const rl = document.createElement('div');
          rl.className = 'c-tool ' + (d.success?'ok':'err');
          rl.textContent = (d.success?'OK ':'FAIL ') + (d.tool||'') + ': ' + (d.text||'').slice(0,80);
          body.appendChild(rl); body.scrollTop = body.scrollHeight;
        }
        else if(d.type === 'done'){
          status.textContent = d.success ? '完成' : '失败';
          status.style.color = d.success ? '#065f46' : '#991b1b';
        }
      }
      scrollDown();
    }
  }

  function scrollDown() { cleanupEmptyMsgs(); const m = document.getElementById('messages'); m.scrollTop = m.scrollHeight; }
  function cleanupEmptyMsgs() { document.querySelectorAll('.msg').forEach(el=>{ const txt=el.querySelector('.msg-text'); if(txt && !txt.innerHTML.trim()) el.remove(); }); }
  function onKey(e) {
    var isEnter = e.key === 'Enter' || e.keyCode === 13;
    if(isEnter && !e.shiftKey && !e.isComposing){
      e.preventDefault();
      send();
    }
  }

  function send() {
    const input = document.getElementById('input'); const text = input.value.trim();
    if(!text) return;
    if(!ws || !wsReady || ws.readyState!==WebSocket.OPEN){ toast('连接已断开，正在重连...'); connect(); return; }
    document.getElementById('welcome')?.remove();
    const div = document.createElement('div'); div.className = 'msg user';
    div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + renderMD(text) + '</div></div>';
    document.getElementById('msgList').appendChild(div); scrollDown();
    input.value = ''; input.style.height = 'auto';
    document.getElementById('sendBtn').disabled = true;
    ws.send(JSON.stringify({message:text,session_id:sessionId}));
  }
  function continueStage() {
    const awaitMsg = document.getElementById('awaitConfirmMsg');
    if(awaitMsg) awaitMsg.remove();
    if(ws && wsReady){
      ws.send(JSON.stringify({message:'继续',session_id:sessionId,action:'continue_stage'}));
    }
    document.getElementById('sendBtn').disabled = false;
  }


  async function newChat() {
    try{ const r = await fetch('/api/sessions',{method:'POST'}); const d = await r.json(); sessionId = d.session_id; document.getElementById('msgList').innerHTML = '<div class=\"welcome\" id=\"welcome\"><div class=\"welcome-icon\">&#9670;</div><h2>有什么我可以帮你的？</h2><p style=\"color:var(--text-dim)\">我可以操控电脑、写代码、查资料、管理项目</p></div>'; document.getElementById('msgCount').textContent = '就绪'; agentCards = {}; }catch(e){ toast('创建会话失败'); }
  }

  document.getElementById('input').addEventListener('input',function(){ this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight,140)+'px'; });
  document.getElementById('input').addEventListener('keydown',function(e){
    if((e.key==='Enter'||e.keyCode===13) && !e.shiftKey && !e.isComposing){ e.preventDefault(); send(); }
  });
  init();
</script>
</body>
</html>"""


async def main():
    from soul.config.manager import ConfigManager
    cfg_mgr = ConfigManager()
    config = cfg_mgr.load()
    agent = Agent(config=config)
    await agent.initialize()
    gateway = Gateway(config.gateway)
    await gateway.start(agent, config.gateway.host, config.gateway.port)
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        await gateway.stop()


if __name__ == "__main__":
    asyncio.run(main())
