"""Web UI 入口 — 等同于 soul gateway。

保留此文件仅为兼容旧用法: python -m web.app
实际推荐直接使用: soul gateway
"""

import asyncio
from soul.gateway.server import main

if __name__ == "__main__":
    asyncio.run(main())
