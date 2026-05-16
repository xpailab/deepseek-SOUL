"""Cron 定时任务调度器。

支持:
- 自然语言描述任务("每天早上 9 点发日报")
- 标准 cron 表达式
- 多平台结果投递
- 独立执行 lane（不阻塞主回复）
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from croniter import croniter


@dataclass
class ScheduledTask:
    """定时任务定义。"""
    id: str
    name: str
    description: str = ""
    cron_expr: str = ""  # 标准 5 字段 cron 表达式
    natural_language: str = ""  # 自然语言描述（如"每天早上9点"）
    handler: Callable[..., Coroutine[Any, Any, Any]] | None = None
    enabled: bool = True
    last_run: float = 0
    next_run: float = 0
    run_count: int = 0
    error_count: int = 0
    max_runs: int = 0  # 0 = 无限制
    deliver_to: str = ""  # 结果投递目标通道
    metadata: dict[str, Any] = field(default_factory=dict)


class CronScheduler:
    """定时任务调度器。

    独立执行 lane，不阻塞主入站回复。
    """

    def __init__(self, max_concurrent: int = 2):
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._main_loop: asyncio.Task | None = None

    def add_task(self, task: ScheduledTask) -> None:
        """添加定时任务。"""
        if task.cron_expr:
            # 计算下次运行时间
            base = time.time()
            cron = croniter(task.cron_expr, base)
            task.next_run = cron.get_next(start_time=base)
        self._tasks[task.id] = task

    def add_from_natural_language(
        self,
        name: str,
        description: str,
        natural_lang: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> ScheduledTask:
        """从自然语言描述创建定时任务。

        例如:
            "每天早上9点" → 0 9 * * *
            "每小时"     → 0 * * * *
            "每周一10点"  → 0 10 * * 1
        """
        cron_expr = self._parse_natural_cron(natural_lang)

        task = ScheduledTask(
            id=f"cron_{int(time.time())}_{name}",
            name=name,
            description=description,
            cron_expr=cron_expr,
            natural_language=natural_lang,
            handler=handler,
        )
        self.add_task(task)
        return task

    def remove_task(self, task_id: str) -> bool:
        return self._tasks.pop(task_id, None) is not None

    async def start(self) -> None:
        """启动调度器。"""
        self._running = True
        self._main_loop = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        if self._main_loop:
            self._main_loop.cancel()
            try:
                await self._main_loop
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """主调度循环 — 每秒检查一次。"""
        while self._running:
            now = time.time()
            due_tasks = []

            for task in self._tasks.values():
                if not task.enabled:
                    continue
                if task.max_runs > 0 and task.run_count >= task.max_runs:
                    continue
                if task.next_run > 0 and now >= task.next_run:
                    due_tasks.append(task)

            # 并行执行到期任务（受最大并发限制）
            for task in due_tasks:
                asyncio.create_task(self._execute_task(task))

            await asyncio.sleep(1)

    async def _execute_task(self, task: ScheduledTask) -> None:
        """执行单个定时任务。"""
        async with self._semaphore:
            try:
                task.last_run = time.time()

                if task.handler:
                    await task.handler()

                task.run_count += 1

                # 计算下次运行时间
                if task.cron_expr:
                    cron = croniter(task.cron_expr, task.last_run)
                    task.next_run = cron.get_next(start_time=task.last_run)

            except Exception as e:
                task.error_count += 1
                # 错误时仍然计算下次运行时间
                if task.cron_expr:
                    cron = croniter(task.cron_expr, task.last_run)
                    task.next_run = cron.get_next(start_time=task.last_run)

    @staticmethod
    def _parse_natural_cron(natural: str) -> str:
        """从自然语言解析 cron 表达式。"""
        text = natural.lower().strip()

        # 每天
        if "每天" in text or "every day" in text:
            hour = 9
            if match := re.search(r'(\d{1,2})\s*点', text):
                hour = int(match.group(1))
            return f"0 {hour} * * *"

        # 每小时
        if "每小时" in text or "every hour" in text:
            return "0 * * * *"

        # 每周一
        if "每周一" in text or "every monday" in text:
            hour = 10
            if match := re.search(r'(\d{1,2})\s*点', text):
                hour = int(match.group(1))
            return f"0 {hour} * * 1"

        # 每 N 分钟
        if match := re.search(r'每\s*(\d+)\s*分', text):
            mins = int(match.group(1))
            return f"*/{mins} * * * *"

        # 默认：每天早上 9 点
        return "0 9 * * *"

    def list_tasks(self) -> list[dict[str, Any]]:
        """列出所有定时任务。"""
        return [
            {
                "id": t.id,
                "name": t.name,
                "cron": t.cron_expr,
                "natural": t.natural_language,
                "enabled": t.enabled,
                "next_run": t.next_run,
                "run_count": t.run_count,
                "error_count": t.error_count,
                "last_run": t.last_run,
            }
            for t in self._tasks.values()
        ]
