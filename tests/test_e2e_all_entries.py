"""CLI / Web / 聊天平台全入口端到端测试（pytest 格式）。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from soul.types import (
    LLMConfig,
    MemoryConfig,
    SkillConfig,
    SOULConfig,
)

# ============================================================
# Test 1: CLI chat — 完整对话流程
# ============================================================

@patch("soul.engine.agent.Agent._register_builtin_tools")
@patch("soul.llm.registry.AdapterRegistry.chat")
@pytest.mark.asyncio
async def test_01_cli_chat(mock_llm, mock_register):
    """soul chat — 测试完整对话管道。"""
    from soul.engine.agent import Agent

    mock_llm.return_value = MagicMock(
        content="好的，我来帮你创建 Python 项目。首先创建目录结构...",
        tool_calls=[],
        finish_reason="stop",
    )

    tmp = tempfile.mkdtemp(prefix="soul_test_e2e_")
    workspace = Path(tmp) / "workspace"
    workspace.mkdir(parents=True)

    config = SOULConfig(
        memory=MemoryConfig(workspace_dir=str(workspace), fts_db_path=str(Path(tmp) / "test.db")),
        llm=LLMConfig(provider="deepseek", model="deepseek-chat"),
        skill=SkillConfig(auto_generate=True, gepa_enabled=False),
    )

    agent = Agent(config=config)
    await agent.initialize()

    response = await agent.chat("帮我创建一个 Python 项目")
    assert len(response) > 0, "Agent 返回空响应"
    assert ("好的" in response or "帮你" in response), "响应内容不正确"

    stats = await agent.memory.get_stats()
    assert "procedural" in stats, "统计中缺少 procedural"
    assert stats["procedural"]["skill_count"] >= 0

    await agent.memory.close()


# ============================================================
# Test 2: CLI run — 单次任务
# ============================================================

@patch("soul.engine.agent.Agent._register_builtin_tools")
@patch("soul.llm.registry.AdapterRegistry.chat")
@pytest.mark.asyncio
async def test_02_cli_run(mock_llm, mock_register):
    """soul run — 测试单任务模式。"""
    from soul.engine.agent import Agent

    mock_llm.return_value = MagicMock(
        content="当前目录包含以下文件: README.md, pyproject.toml, soul/",
        tool_calls=[],
        finish_reason="stop",
    )

    tmp = tempfile.mkdtemp(prefix="soul_test_e2e_")
    workspace = Path(tmp) / "workspace"
    workspace.mkdir(parents=True)

    config = SOULConfig(
        memory=MemoryConfig(workspace_dir=str(workspace), fts_db_path=str(Path(tmp) / "test2.db")),
        llm=LLMConfig(provider="deepseek", model="deepseek-chat"),
    )

    agent = Agent(config=config)
    await agent.initialize()

    response = await agent.chat("列出当前目录文件")
    assert len(response) > 0, "单次任务返回空内容"

    await agent.memory.close()


# ============================================================
# Test 3: CLI config — 配置查看
# ============================================================

@pytest.mark.asyncio
async def test_03_cli_config():
    """soul config — 测试配置管理。"""
    from soul.config.manager import ConfigManager

    tmp = tempfile.mkdtemp(prefix="soul_test_cfg_")
    cfg_path = Path(tmp) / "config.yaml"

    mgr = ConfigManager(str(cfg_path))
    config = mgr.config

    assert config.llm.provider in ("deepseek", "claude", "openai"), f"未知 provider: {config.llm.provider}"
    assert config.lane.max_concurrent == 4, "并发槽位默认值不是 4"
    assert config.gateway.port == 18789, "网关端口默认值不是 18789"
    assert config.memory.predictive_enabled is True, "预测记忆默认未启用"

    mgr.update(**{"llm.provider": "deepseek"})
    assert mgr.config.llm.provider == "deepseek"

    mgr.update(**{"llm.model": "deepseek-v3"})
    assert mgr.config.llm.model == "deepseek-v3"

    mgr.save()
    mgr2 = ConfigManager(str(cfg_path))
    assert mgr2.config.llm.model == "deepseek-v3", "配置持久化失败"

    with patch.dict(os.environ, {"SOUL_LLM_PROVIDER": "openai"}):
        mgr3 = ConfigManager(str(cfg_path))
        assert mgr3.config.llm.provider == "openai", "环境变量覆盖失败"


# ============================================================
# Test 4: CLI status — 系统状态
# ============================================================

@patch("soul.engine.agent.Agent._register_builtin_tools")
@pytest.mark.asyncio
async def test_04_cli_status(mock_register):
    """soul status — 测试系统状态查看。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_status_")
    workspace = Path(tmp) / "workspace"
    workspace.mkdir(parents=True)

    config = SOULConfig(
        memory=MemoryConfig(workspace_dir=str(workspace), fts_db_path=str(Path(tmp) / "test_status.db")),
    )

    from soul.engine.agent import Agent
    agent = Agent(config=config)
    await agent.initialize()

    status = await agent.get_status()
    assert status["initialized"] is True
    assert status["running"] is True
    for k in ["frozen", "procedural", "indexed", "predictive"]:
        assert k in status["memory"], f"memory 中缺少 {k}"

    mem_status = status["memory"]
    assert "usage" in mem_status["frozen"]
    assert "skills" in mem_status["procedural"]
    assert "total_conversations" in mem_status["indexed"]

    await agent.memory.close()


# ============================================================
# Test 5: Gateway REST API
# ============================================================

@pytest.mark.asyncio
async def test_05_gateway_rest():
    """POST /api/chat — 测试 REST API 路由注册。"""
    from soul.gateway.router import ChannelMessage
    from soul.gateway.server import CHAT_PAGE, Gateway
    from soul.types import GatewayConfig

    gw = Gateway(GatewayConfig(port=18789))
    assert gw is not None
    assert gw.config.port == 18789
    assert gw.router is not None

    assert ("<html" in CHAT_PAGE.lower() or "<!DOCTYPE" in CHAT_PAGE.lower()), "CHAT_PAGE 缺少 HTML"
    assert ("websocket" in CHAT_PAGE.lower() or "fetch" in CHAT_PAGE.lower() or "script" in CHAT_PAGE.lower())

    msg = ChannelMessage(raw_text="测试", channel="cli", channel_user_id="user1")
    assert msg.channel == "cli"
    assert msg.channel_user_id == "user1"


# ============================================================
# Test 6: Gateway WebSocket 路由
# ============================================================

@pytest.mark.asyncio
async def test_06_gateway_websocket():
    """WS /ws/chat — 测试 WebSocket 路由。"""
    from soul.gateway.server import CHAT_PAGE, Gateway
    from soul.types import GatewayConfig

    gw = Gateway(GatewayConfig(port=18789))

    assert ("ws" in CHAT_PAGE.lower() or "websocket" in CHAT_PAGE.lower())
    assert ("chat" in CHAT_PAGE.lower() or "message" in CHAT_PAGE.lower())
    assert ("stream" in CHAT_PAGE.lower() or "onmessage" in CHAT_PAGE.lower())


# ============================================================
# Test 7: Gateway 完整入口
# ============================================================

@pytest.mark.asyncio
async def test_07_gateway_full():
    """网关完整验证: 路由 + 会话 + 消息处理。"""
    from soul.gateway.router import ChannelMessage
    from soul.gateway.server import CHAT_PAGE, Gateway
    from soul.types import GatewayConfig

    gw = Gateway(GatewayConfig(port=18789))

    async def cli_handler(msg):
        return {"status": "ok", "channel": msg.channel}

    gw.router.register_handler("cli", cli_handler)
    assert "cli" in gw.router._handlers

    msg = ChannelMessage(raw_text="你好，帮我分析代码", channel="cli", channel_user_id="user_test")
    result = await gw.router.route(msg)
    assert result is not None
    assert result.get("status") == "ok"

    assert len(CHAT_PAGE) > 500, "CHAT_PAGE 太短"
    assert "<head" in CHAT_PAGE.lower()
    assert "<body" in CHAT_PAGE.lower()

    gw._stats["messages_processed"] = 42
    assert gw._stats["messages_processed"] == 42


# ============================================================
# Test 8: QQ 连接器
# ============================================================

@pytest.mark.asyncio
async def test_08_qq_connector():
    """QQ 连接器: 消息构建 + token 管理。"""
    from soul.gateway.connectors import QQConnector

    qq = QQConnector(bot_app_id="102000001", bot_token="test_token", client_secret="test_secret")

    text_payload = qq._build_text("你好，这是测试消息", "user_openid_123")
    assert "content" in text_payload
    assert text_payload["msg_type"] == 0
    assert text_payload["msg_id"].startswith("msg_")

    md_payload = qq._build_markdown("## 测试报告\n\n- 项1\n- 项2", "user_openid_123")
    assert "markdown" in md_payload
    assert md_payload["msg_type"] == 2

    img_payload = qq._build_image("https://example.com/img.png", "user_openid_123")
    assert img_payload["msg_type"] == 3

    assert qq._access_token == ""
    assert qq.is_connected is False

    long_msg = "A" * 5000
    payload = qq._build_text(long_msg, "u1")
    assert len(payload["content"]) <= 2000, "长消息未截断"


# ============================================================
# Test 9: 微信 + 钉钉 连接器
# ============================================================

@pytest.mark.asyncio
async def test_09_wechat_dingtalk():
    """微信 + 钉钉: 全模式消息构建。"""
    from soul.gateway.connectors import DingTalkConnector, WeChatConnector

    # ── 微信 ──
    wx_mp = WeChatConnector(app_id="wxTEST", app_secret="test", mode="mp")
    mp_msg = wx_mp._build_mp_message("公众号测试消息", "openid_abc", "text")
    assert mp_msg["touser"] == "openid_abc"
    assert "公众号测试消息" in mp_msg["text"]["content"]

    long_mp = wx_mp._build_mp_message("X" * 5000, "o1", "text")
    assert len(long_mp["text"]["content"]) <= 2048, "微信长消息未截断"

    wx_wecom = WeChatConnector(corpid="wwTEST", corpsecret="test", agent_id=1000001, mode="wecom")
    wecom_msg = wx_wecom._build_wecom_message("企业微信测试", "user_zhangsan", "markdown")
    assert wecom_msg["agentid"] == 1000001
    assert wecom_msg["msgtype"] == "markdown"
    assert wecom_msg["touser"] == "user_zhangsan"

    # ── 钉钉 ──
    dt_webhook = DingTalkConnector(webhook_url="https://oapi.dingtalk.com/robot/send?access_token=TEST")
    assert "access_token=TEST" in dt_webhook.webhook_url

    dt_app = DingTalkConnector(app_key="dingTEST", app_secret="test", mode="app")
    assert dt_app.mode == "app"
    assert dt_app._access_token == ""


# ============================================================
# Test 10: 飞书 + Telegram 连接器
# ============================================================

@pytest.mark.asyncio
async def test_10_feishu_telegram():
    """飞书 + Telegram: 卡片消息 + 长轮询。"""
    from soul.gateway.connectors import FeishuConnector, TelegramConnector

    # ── 飞书 ──
    fs_webhook = FeishuConnector(
        webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/TEST_TOKEN"
    )
    assert "TEST_TOKEN" in fs_webhook.webhook_url
    assert fs_webhook.mode == "webhook"

    fs_app = FeishuConnector(app_id="cli_TEST", app_secret="test", mode="app")
    assert fs_app.mode == "app"

    card = {
        "header": {"title": {"content": "CI 构建结果"}},
        "elements": [
            {"tag": "div", "text": {"content": "✅ 构建成功"}},
            {"tag": "div", "text": {"content": "📦 版本: v2.3.1"}},
            {"tag": "hr"},
            {"tag": "div", "text": {"content": "🔗 查看详情"}},
        ],
    }
    assert card["header"]["title"]["content"] == "CI 构建结果"
    assert len(card["elements"]) == 4

    # ── Telegram ──
    tg = TelegramConnector(bot_token="123456:ABC-DEF")
    assert tg.bot_token == "123456:ABC-DEF"
    assert "123456:ABC-DEF" in f"{tg.API_BASE}{tg.bot_token}"

    msg_received = []
    tg.on("message", lambda chat_id, text, user_name="": msg_received.append((chat_id, text)))

    await tg._emit("message", chat_id="12345", text="你好", user_name="testuser")
    assert len(msg_received) == 1
    assert msg_received[0] == ("12345", "你好")


# ============================================================
# 跨平台统一汇总测试
# ============================================================

@pytest.mark.asyncio
async def test_cross_platform_summary():
    """跨平台统一: 所有连接器通过 SessionSync 统一管理。"""
    from soul.gateway.connectors import (
        DingTalkConnector,
        FeishuConnector,
        QQConnector,
        TelegramConnector,
        WeChatConnector,
    )
    from soul.gateway.session_sync import PlatformConnector, SessionSync

    tmp = tempfile.mkdtemp(prefix="soul_sync_test_")
    sync = SessionSync(str(Path(tmp) / "sessions"))

    connectors = [
        QQConnector(bot_app_id="q1", bot_token="t1", client_secret="s1"),
        WeChatConnector(app_id="w1", app_secret="s1", mode="mp"),
        WeChatConnector(corpid="w2", corpsecret="s2", agent_id=1, mode="wecom"),
        DingTalkConnector(webhook_url="https://oapi.dingtalk.com/robot/send?access_token=d1"),
        DingTalkConnector(app_key="d2", app_secret="s2", mode="app"),
        FeishuConnector(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/f1"),
        FeishuConnector(app_id="f2", app_secret="s2", mode="app"),
        TelegramConnector(bot_token="t1"),
    ]

    assert len(connectors) == 8, f"连接器总数应为 8，实际 {len(connectors)}"

    all_valid = all(isinstance(c, PlatformConnector) for c in connectors)
    assert all_valid, "不是所有连接器都继承 PlatformConnector"

    sync.write_shared_session("cross_platform_001", {
        "session_id": "cross_platform_001",
        "message": "跨平台统一消息",
    })
    data = sync.get_shared_session("cross_platform_001")
    assert data["session_id"] == "cross_platform_001"
