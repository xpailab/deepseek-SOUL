"""消息网关 — 多平台统一接入。"""
from soul.gateway.router import MessageRouter
from soul.gateway.server import Gateway

__all__ = ["Gateway", "MessageRouter"]
