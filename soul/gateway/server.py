"""Gateway 服务器 — 中央控制平面。

基于 FastAPI + WebSocket，统一接入多平台消息。
管理所有 sessions、channels、events。

启动:
    soul-gateway --port 18789
    python -m soul.gateway.server
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from soul.engine.agent import Agent
from soul.gateway.router import ChannelMessage, MessageRouter
from soul.types import GatewayConfig, QueueMode


class Gateway:
    """统一消息网关。

    职责:
    - 管理所有消息通道
    - 路由消息到 Agent
    - WebSocket 配对客户端
    - 会话状态管理
    - 模型路由与切换
    """

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig()
        self.router = MessageRouter()
        self.agent: Agent | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._stats: dict[str, Any] = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": 0,
        }

    async def start(
        self,
        agent: Agent | None = None,
        host: str = "",
        port: int = 0,
    ) -> None:
        """启动网关。

        可选的 FastAPI 集成:
            from fastapi import FastAPI
            app = FastAPI()
            gateway = Gateway()
            await gateway.start(agent)

            @app.post("/chat")
            async def chat(request: ChatRequest):
                return await gateway.handle_message(...)
        """
        self.agent = agent
        self._running = True
        self._stats["uptime_start"] = time.time()

        host = host or self.config.host
        port = port or self.config.port

        # 在后台启动 WebSocket 服务器
        if self.config.websocket_enabled:
            self._tasks.append(
                asyncio.create_task(self._run_ws_server(host, port))
            )

        print(f"[Gateway] 网关已启动 — ws://{host}:{port}")

        # 注册默认处理器
        self.router.register_handler("cli", self._handle_cli_message)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.agent:
            await self.agent.shutdown()

    async def handle_message(self, msg: ChannelMessage) -> dict[str, Any]:
        """处理来自任意通道的消息。"""
        if not self.agent:
            return {"error": "Agent 未初始化"}

        self._stats["messages_processed"] += 1

        try:
            # DM 安全：检查配对
            if msg.is_dm and self.config.dm_policy == "pairing":
                if not self._check_pairing(msg):
                    return {"reply": "请先完成设备配对。使用 'soul pairing approve <code>' 进行配对。"}

            # 入队并处理
            queue_item = msg.to_message()
            queue_mode = msg.resolve_queue_mode(msg.raw_text)

            # 使用 Agent 处理
            response = await self.agent.chat(
                msg.raw_text,
                session_id=msg.session_id,
            )

            return {"reply": response, "session_id": msg.session_id}

        except Exception as e:
            self._stats["errors"] += 1
            return {"error": str(e)}

    async def handle_stream(self, msg: ChannelMessage):
        """处理流式消息（返回异步迭代器）。"""
        if not self.agent:
            yield {"error": "Agent 未初始化"}
            return

        async for chunk in self.agent.chat_stream(
            msg.raw_text,
            session_id=msg.session_id,
        ):
            yield chunk

    def _check_pairing(self, msg: ChannelMessage) -> bool:
        """检查设备配对状态。"""
        # 简化实现：默认允许
        return True

    async def _handle_cli_message(self, msg: ChannelMessage) -> dict[str, Any]:
        """处理 CLI 通道消息。"""
        return await self.handle_message(msg)

    async def _run_ws_server(self, host: str, port: int) -> None:
        """运行 WebSocket 服务器。"""
        try:
            import uvicorn
            from fastapi import FastAPI, WebSocket

            app = FastAPI(title="DeepSoul Gateway")

            @app.get("/health")
            async def health():
                return {"status": "ok", "uptime": time.time() - self._stats["uptime_start"]}

            @app.websocket("/ws/{client_id}")
            async def ws_endpoint(websocket: WebSocket, client_id: str):
                await websocket.accept()
                try:
                    while self._running:
                        data = await websocket.receive_text()
                        msg = ChannelMessage(
                            raw_text=data,
                            channel="websocket",
                            channel_user_id=client_id,
                        )
                        result = await self.handle_message(msg)
                        await websocket.send_json(result)
                except Exception:
                    pass

            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

        except ImportError:
            # FastAPI/uvicorn 未安装，降级为简单 socket
            import asyncio

            async def handle_client(reader, writer):
                try:
                    while self._running:
                        data = await reader.read(4096)
                        if not data:
                            break
                        text = data.decode("utf-8").strip()
                        msg = ChannelMessage(
                            raw_text=text,
                            channel="tcp",
                            channel_user_id=f"{writer.get_extra_info('peername')}",
                        )
                        result = await self.handle_message(msg)
                        writer.write(str(result).encode("utf-8") + b"\n")
                        await writer.drain()
                except Exception:
                    pass
                finally:
                    writer.close()

            server = await asyncio.start_server(handle_client, host, port)
            print(f"[Gateway] TCP 服务器运行中 — {host}:{port}")
            async with server:
                await server.serve_forever()

    def get_stats(self) -> dict[str, Any]:
        stats = dict(self._stats)
        stats["uptime"] = time.time() - stats["uptime_start"] if stats["uptime_start"] else 0
        if self.agent:
            stats["agent"] = {
                "llm_provider": self.agent.config.llm.provider,
                "llm_model": self.agent.config.llm.model,
            }
        return stats


async def main():
    """soul-gateway 入口。"""
    from soul.config.manager import ConfigManager

    cfg_mgr = ConfigManager()
    config = cfg_mgr.load()

    agent = Agent(config=config)
    await agent.initialize()

    gateway = Gateway(config.gateway)
    await gateway.start(agent, config.gateway.host, config.gateway.port)

    try:
        print(f"\nDeepSoul Gateway 运行中...")
        print(f"端口: {config.gateway.port}")
        print(f"按 Ctrl+C 停止\n")
        # 保持运行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        await gateway.stop()


if __name__ == "__main__":
    asyncio.run(main())
