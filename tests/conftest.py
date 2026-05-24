"""Shared pytest fixtures for DeepSoul tests."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from soul.types import (
    GatewayConfig,
    LLMConfig,
    MemoryConfig,
    SandboxConfig,
    SkillConfig,
    SOULConfig,
)


@pytest.fixture
def tmp_workspace() -> Any:
    """创建临时工作区，测试后自动清理。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_")
    workspace = Path(tmp) / "workspace"
    workspace.mkdir(parents=True)
    yield workspace
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def base_config(tmp_workspace: Path) -> SOULConfig:
    """创建最小化 SOULConfig 用于测试。"""
    return SOULConfig(
        memory=MemoryConfig(
            workspace_dir=str(tmp_workspace),
            fts_db_path=str(tmp_workspace / "test.db"),
        ),
        llm=LLMConfig(provider="deepseek", model="deepseek-chat"),
        skill=SkillConfig(auto_generate=False, gepa_enabled=False),
    )


@pytest.fixture
def gateway_config() -> GatewayConfig:
    """GatewayConfig 测试用。"""
    return GatewayConfig(port=18789)


@pytest.fixture
def sandbox_config() -> SandboxConfig:
    """SandboxConfig 测试用。"""
    return SandboxConfig(default_mode="local", readonly_root=True)


@pytest.fixture
def event_loop() -> Any:
    """为异步测试创建独立的事件循环。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_adapter() -> Any:
    """创建 mock LLM 适配器。"""
    adapter = MagicMock()
    adapter.chat = AsyncMock(return_value=MagicMock(
        content="mock response",
        tool_calls=[],
        finish_reason="stop",
        reasoning_content="",
    ))
    adapter.chat_stream = AsyncMock(return_value=[])

    async def _stream():
        yield MagicMock(content="mock ", tool_calls=[], finish_reason="")
        yield MagicMock(content="response", tool_calls=[], finish_reason="stop")

    adapter.chat_stream = _stream
    return adapter
