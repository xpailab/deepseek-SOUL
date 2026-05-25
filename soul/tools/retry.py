"""工具重试与速率限制管理器。

提供智能重试策略和 API 速率限制追踪。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any


class RateLimitTracker:
    """API 速率限制追踪器。

    基于滑动窗口的速率限制，支持多 API 提供商。
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        """检查是否可以发送请求。"""
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False

    async def wait_until_ready(self, timeout: float = 120.0) -> bool:
        """等待直到可以发送请求。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.check():
                return True
            await asyncio.sleep(1.0)
        return False

    @property
    def available_tokens(self) -> int:
        now = time.monotonic()
        cutoff = now - self.window
        active = sum(1 for t in self._timestamps if t > cutoff)
        return max(0, self.max_requests - active)


class RetryManager:
    """智能重试管理器。

    支持策略:
    - exponential_backoff: 指数退避
    - linear: 线性等待
    - immediate: 立即重试
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._tracker = RateLimitTracker()

    async def execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        tool_name: str = "",
        **kwargs: Any,
    ) -> tuple[Any, str | None, int]:
        """执行函数并带重试。

        Returns:
            (result, error, retry_count)
        """
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                # 速率限制检查
                if not await self._tracker.check():
                    delay = self._calc_delay(attempt)
                    await asyncio.sleep(delay)
                    continue

                result = await func(*args, **kwargs)
                return result, None, attempt

            except TimeoutError:
                last_error = f"{tool_name}: 执行超时"
                if attempt < self.max_retries:
                    delay = self._calc_delay(attempt)
                    await asyncio.sleep(delay)
                continue

            except PermissionError as e:
                last_error = f"{tool_name}: 权限不足 - {e}"
                break  # 权限错误不重试

            except FileNotFoundError as e:
                last_error = f"{tool_name}: 文件不存在 - {e}"
                break  # 文件不存在不重试

            except Exception as e:
                last_error = f"{tool_name}: {e}"
                if attempt < self.max_retries:
                    delay = self._calc_delay(attempt)
                    await asyncio.sleep(delay)
                continue

        return None, last_error, self.max_retries

    def _calc_delay(self, attempt: int) -> float:
        """计算退避延迟（指数退避 + 随机抖动）。"""
        import random
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random()
        return delay

    @property
    def rate_limiter(self) -> RateLimitTracker:
        return self._tracker
