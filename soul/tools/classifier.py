"""工具结果分类器 — 将工具执行结果分为不同类别。

分类:
- success: 完全成功
- partial: 部分成功
- denied: 权限拒绝
- failure: 执行失败
- timeout: 超时
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from soul.types import ToolResult


class ResultCategory(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    DENIED = "denied"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class ResultClassifier:
    """工具结果分类器。

    根据执行结果的特征自动分类，指导后续处理策略。
    """

    # 错误关键词 → 分类映射
    ERROR_PATTERNS: dict[str, ResultCategory] = {
        "permission denied": ResultCategory.DENIED,
        "access denied": ResultCategory.DENIED,
        "not authorized": ResultCategory.DENIED,
        "forbidden": ResultCategory.DENIED,
        "eacces": ResultCategory.DENIED,
        "eperm": ResultCategory.DENIED,

        "timeout": ResultCategory.TIMEOUT,
        "timed out": ResultCategory.TIMEOUT,
        "deadline exceeded": ResultCategory.TIMEOUT,
        "connection reset": ResultCategory.TIMEOUT,

        "rate limit": ResultCategory.RATE_LIMITED,
        "too many requests": ResultCategory.RATE_LIMITED,
        "429": ResultCategory.RATE_LIMITED,

        "not found": ResultCategory.PARTIAL,
        "no such file": ResultCategory.PARTIAL,
        "does not exist": ResultCategory.PARTIAL,
    }

    def classify(
        self,
        tool_name: str,
        result: Any,
        error: str | None = None,
        duration_ms: float = 0,
        timeout_seconds: float = 60.0,
    ) -> ToolResult:
        """分类工具执行结果。

        Args:
            tool_name: 工具名称
            result: 执行结果
            error: 错误信息
            duration_ms: 执行耗时
            timeout_seconds: 超时阈值

        Returns:
            ToolResult with classification
        """
        call_id = f"tc_{int(time.time() * 1000)}"

        # 超时检测
        if duration_ms > timeout_seconds * 1000:
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                success=False,
                result=result,
                error="执行超时",
                duration_ms=duration_ms,
                classification=ResultCategory.TIMEOUT.value,
            )

        # 错误分类
        if error:
            category = self._classify_error(error)
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                success=False,
                result=result,
                error=error,
                duration_ms=duration_ms,
                classification=category.value,
            )

        # 成功
        return ToolResult(
            call_id=call_id,
            name=tool_name,
            success=True,
            result=result,
            duration_ms=duration_ms,
            classification=ResultCategory.SUCCESS.value,
        )

    def _classify_error(self, error: str) -> ResultCategory:
        """根据错误信息分类。"""
        error_lower = error.lower()
        for pattern, category in self.ERROR_PATTERNS.items():
            if pattern in error_lower:
                return category
        return ResultCategory.FAILURE

    def get_retry_strategy(self, classification: str) -> str:
        """根据分类获取重试策略。"""
        strategies = {
            ResultCategory.TIMEOUT.value: "exponential_backoff",
            ResultCategory.RATE_LIMITED.value: "wait_and_retry",
            ResultCategory.PARTIAL.value: "rephrase_and_retry",
            ResultCategory.DENIED.value: "ask_permission",
            ResultCategory.FAILURE.value: "analyze_and_retry",
            ResultCategory.SUCCESS.value: "none",
        }
        return strategies.get(classification, "none")

    def is_retryable(self, classification: str) -> bool:
        """判断是否可重试。"""
        retryable = {
            ResultCategory.TIMEOUT.value,
            ResultCategory.RATE_LIMITED.value,
            ResultCategory.PARTIAL.value,
        }
        return classification in retryable
