"""工具注册中心 — 统一管理工具的定义、注册和查询。

所有工具在此注册，Agent 通过此获取可用工具列表。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from soul.types import ToolRisk

# 工具回调类型
ToolHandler = Callable[..., Coroutine[Any, Any, Any]]


class ToolDef:
    """工具定义。"""

    def __init__(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        parameters: dict[str, Any] | None = None,
        risk: ToolRisk = ToolRisk.SAFE,
        requires_approval: bool = False,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        sandbox_only: bool = False,
        tags: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters = parameters or {"type": "object", "properties": {}, "required": []}
        self.risk = risk
        self.requires_approval = requires_approval
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sandbox_only = sandbox_only
        self.tags = tags or []
        self.call_count = 0
        self.error_count = 0

    def to_api_schema(self) -> dict[str, Any]:
        """转为 OpenAI function calling 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def success_rate(self) -> float:
        total = self.call_count
        if total == 0:
            return 1.0
        return (total - self.error_count) / total


class ToolRegistry:
    """工具注册中心。

    管理所有可用工具，提供注册、查找、列表功能。
    """

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._tags_index: dict[str, list[str]] = {}

    def register(self, tool: ToolDef) -> None:
        """注册工具。"""
        self._tools[tool.name] = tool
        for tag in tool.tags:
            self._tags_index.setdefault(tag, []).append(tool.name)

    def register_many(self, tools: list[ToolDef]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def list_by_risk(self, max_risk: ToolRisk) -> list[ToolDef]:
        """列出不超过指定风险等级的工具。"""
        risk_order = list(ToolRisk)
        max_idx = risk_order.index(max_risk)
        return [
            t for t in self._tools.values()
            if risk_order.index(t.risk) <= max_idx
        ]

    def list_by_tag(self, tag: str) -> list[ToolDef]:
        names = self._tags_index.get(tag, [])
        return [self._tools[n] for n in names if n in self._tools]

    def list_sandbox_safe(self) -> list[ToolDef]:
        """列出沙箱中安全可用的工具。"""
        return [
            t for t in self._tools.values()
            if t.risk in (ToolRisk.SAFE, ToolRisk.LOW)
        ]

    def to_api_schemas(self, max_risk: ToolRisk | None = None) -> list[dict[str, Any]]:
        """生成 API 格式的工具列表。"""
        tools = self.list_all()
        if max_risk:
            tools = self.list_by_risk(max_risk)
        return [t.to_api_schema() for t in tools]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_tools": len(self._tools),
            "by_risk": {
                risk.value: len(self.list_by_risk(risk))
                for risk in ToolRisk
            },
            "total_calls": sum(t.call_count for t in self._tools.values()),
            "total_errors": sum(t.error_count for t in self._tools.values()),
        }

    def unregister(self, name: str) -> bool:
        """移除工具注册。"""
        if name in self._tools:
            tool = self._tools.pop(name)
            for tag in tool.tags:
                if tag in self._tags_index:
                    self._tags_index[tag] = [
                        n for n in self._tags_index[tag] if n != name
                    ]
            return True
        return False
