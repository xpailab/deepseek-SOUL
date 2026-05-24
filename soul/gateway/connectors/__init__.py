"""平台连接器 — QQ / 微信 / 钉钉 / 飞书 / Telegram。"""
from soul.gateway.connectors.base import PlatformConnector
from soul.gateway.connectors.qq import QQConnector
from soul.gateway.connectors.wechat import WeChatConnector
from soul.gateway.connectors.dingtalk import DingTalkConnector
from soul.gateway.connectors.feishu import FeishuConnector
from soul.gateway.connectors.telegram import TelegramConnector

__all__ = [
    "PlatformConnector",
    "QQConnector",
    "WeChatConnector",
    "DingTalkConnector",
    "FeishuConnector",
    "TelegramConnector",
]
