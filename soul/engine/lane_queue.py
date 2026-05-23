"""Lane Queue 2.0 — 双层并发调度系统。

从 OpenClaw 的 Lane Queue 架构改进而来，新增：
- adaptive 队列模式（智能选择最优处理方式）
- 背压控制（系统过载时自动降级）
- 优先级感知调度
- 实时统计与监控

架构:
    Message Arrives
         │
    ┌────▼─────────────────────────────┐
    │  Session Lane (per session)       │
    │  - 并发: 1 (强制串行)             │
    │  - 确保同一会话消息不冲突          │
    │  - 每个 session_id 映射唯一 lane   │
    └────┬─────────────────────────────┘
         │
    ┌────▼─────────────────────────────┐
    │  Global Lane (main)              │
    │  - 并发: maxConcurrent (默认 4)   │
    │  - 限制同时运行的 Agent 总数       │
    │  - 可配置，全局生效               │
    └────┬─────────────────────────────┘
         │
    ┌────▼─────────────────────────────┐
    │  Worker Pool → Agent Instance    │
    └──────────────────────────────────┘

其他 Lane 类型:
- cron: 定时任务
- subagent: 子智能体
- nested: 嵌套流程
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from soul.types import LaneConfig, QueueMode


@dataclass
class QueueItem:
    """队列项。"""
    id: str
    session_id: str
    prompt: str
    mode: QueueMode = QueueMode.QUEUE
    priority: int = 0  # 数字越大优先级越高
    enqueued_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneStats:
    """Lane 统计信息。"""
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_dropped: int = 0
    total_steered: int = 0
    total_interrupted: int = 0
    avg_wait_ms: float = 0
    avg_process_ms: float = 0
    current_queue_size: int = 0
    current_active: int = 0


class SessionLane:
    """会话级 Lane — 每个 session 独立，强制串行。

    确保同一会话的消息按序处理，不会出现竞态。
    """

    def __init__(self, session_id: str, config: LaneConfig):
        self.session_id = session_id
        self.config = config
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=config.cap)
        self._active = False
        self._current_item: QueueItem | None = None
        self._debounce_buf: list[QueueItem] = []
        self._debounce_task: asyncio.Task | None = None
        self.stats = LaneStats()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_item(self) -> QueueItem | None:
        return self._current_item

    async def enqueue(self, item: QueueItem) -> bool:
        """入队。"""
        if self._queue.full():
            if self.config.drop_policy == "new":
                self.stats.total_dropped += 1
                return False
            elif self.config.drop_policy == "old":
                try:
                    self._queue.get_nowait()
                    self.stats.total_dropped += 1
                except asyncio.QueueEmpty:
                    pass
            elif self.config.drop_policy == "summarize":
                self._merge_items(item)
                return True

        await self._queue.put(item)
        self.stats.current_queue_size = self._queue.qsize()
        self.stats.total_enqueued += 1
        return True

    async def dequeue(self) -> QueueItem | None:
        """出队（阻塞）。"""
        try:
            item = await self._queue.get()
            self._current_item = item
            self.stats.current_queue_size = self._queue.qsize()
            self.stats.total_dequeued += 1
            return item
        except asyncio.CancelledError:
            return None

    def try_dequeue_nowait(self) -> QueueItem | None:
        """非阻塞出队。"""
        try:
            item = self._queue.get_nowait()
            self._current_item = item
            self.stats.current_queue_size = self._queue.qsize()
            self.stats.total_dequeued += 1
            return item
        except asyncio.QueueEmpty:
            return None

    def mark_done(self) -> None:
        """标记当前项完成。"""
        if self._current_item:
            elapsed = (time.time() - self._current_item.enqueued_at) * 1000
            n = self.stats.total_dequeued
            self.stats.avg_process_ms = (
                (self.stats.avg_process_ms * (n - 1) + elapsed) / n
                if n > 1 else elapsed
            )
        self._current_item = None
        self._active = False

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def _merge_items(self, new_item: QueueItem) -> None:
        """合并积压消息（collect 模式）。"""
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        merged_prompt = "[合并消息]\n"
        for i, item in enumerate(items):
            merged_prompt += f"---\nQueued #{i + 1}\n{item.prompt}\n"
        merged_prompt += f"---\nQueued #{len(items) + 1}\n{new_item.prompt}\n"

        new_item.prompt = merged_prompt
        self._queue.put_nowait(new_item)
        self.stats.current_queue_size = self._queue.qsize()

    async def start_debounce(self, item: QueueItem) -> None:
        """启动防抖计时器。"""
        self._debounce_buf.append(item)
        if self._debounce_task is None or self._debounce_task.done():
            self._debounce_task = asyncio.create_task(self._debounce_timer())

    async def _debounce_timer(self) -> None:
        """防抖计时器。"""
        await asyncio.sleep(self.config.debounce_ms / 1000)
        if self._debounce_buf:
            if len(self._debounce_buf) == 1:
                item = self._debounce_buf.pop(0)
                await self.enqueue(item)
            else:
                merged = self._debounce_buf[0]
                prompts = [item.prompt for item in self._debounce_buf]
                merged.prompt = "\n---\n".join(
                    f"[Queued #{i + 1}]\n{p}"
                    for i, p in enumerate(prompts)
                )
                await self.enqueue(merged)
            self._debounce_buf.clear()


class GlobalLane:
    """全局 Lane — 限制所有 Agent 的总并发数。"""

    def __init__(self, config: LaneConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._session_lanes: dict[str, SessionLane] = {}
        self._active_count = 0
        self._lock = asyncio.Lock()
        self.stats = LaneStats()

    def get_or_create_session_lane(self, session_id: str) -> SessionLane:
        """获取或创建会话 Lane。"""
        if session_id not in self._session_lanes:
            self._session_lanes[session_id] = SessionLane(session_id, self.config)
        return self._session_lanes[session_id]

    async def acquire(self) -> bool:
        """获取全局执行槽位。"""
        acquired = await self._semaphore.acquire()
        if acquired:
            async with self._lock:
                self._active_count += 1
                self.stats.current_active = self._active_count
        return acquired

    def release(self) -> None:
        """释放全局执行槽位（线程安全）。"""
        self._semaphore.release()
        self._active_count = max(0, self._active_count - 1)
        self.stats.current_active = self._active_count

    def remove_session_lane(self, session_id: str) -> None:
        """移除会话 Lane（清理相关资源）。"""
        lane = self._session_lanes.pop(session_id, None)
        if lane and lane._debounce_task and not lane._debounce_task.done():
            lane._debounce_task.cancel()

    @property
    def active_sessions(self) -> int:
        return len(self._session_lanes)

    @property
    def total_pending(self) -> int:
        return sum(
            lane.stats.current_queue_size
            for lane in self._session_lanes.values()
        )


class LaneQueue:
    """Lane Queue 2.0 — 双层并发调度器。

    整合 Session Lane 和 Global Lane，提供统一的入队/出队接口。

    7 种队列模式:
    - steer: 立即注入当前流式输出
    - followup: 排队等待下一回合
    - collect: 合并积压消息（默认）
    - steer_backlog: 注入 + 保留副本
    - interrupt: 立即中断
    - queue: 标准 FIFO
    - adaptive: 智能选择（SOUL 创新）
    """

    def __init__(self, config: LaneConfig | None = None):
        self.config = config or LaneConfig()
        self.global_lane = GlobalLane(self.config)
        self._subagent_semaphore = asyncio.Semaphore(self.config.subagent_concurrent)
        self._cron_semaphore = asyncio.Semaphore(self.config.cron_concurrent)
        # steer 处理回调: (session_id, text) -> None
        self._steer_callbacks: dict[str, Callable[[str], Coroutine[Any, Any, None]]] = {}
        # interrupt 处理回调
        self._interrupt_callbacks: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}
        # 活跃的 Agent 运行
        self._active_runs: dict[str, Any] = {}

    def track_active(self, session_id: str) -> None:
        """标记 session 正在流式输出。"""
        self._active_runs[session_id] = True

    def untrack_active(self, session_id: str) -> None:
        """取消标记。"""
        self._active_runs.pop(session_id, None)

    def register_steer_callback(
        self, session_id: str, callback: Callable[[str], Coroutine[Any, Any, None]]
    ) -> None:
        """注册 steer 回调。"""
        self._steer_callbacks[session_id] = callback

    def register_interrupt_callback(
        self, session_id: str, callback: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        """注册 interrupt 回调。"""
        self._interrupt_callbacks[session_id] = callback

    def resolve_mode(
        self, item: QueueItem, is_streaming: bool, session_busy: bool
    ) -> QueueMode:
        """解析最终队列模式。

        如果模式是 adaptive，根据当前状态智能选择。
        """
        mode = item.mode
        if mode == QueueMode.ADAPTIVE:
            mode = self._adaptive_choice(item, is_streaming, session_busy)
        return mode

    async def enqueue(self, item: QueueItem) -> str:
        """入队消息。

        Returns:
            处理决策: "steered" | "interrupted" | "queued" | "collected"
        """
        session_lane = self.global_lane.get_or_create_session_lane(item.session_id)
        is_streaming = item.session_id in self._active_runs
        is_busy = session_lane.is_active

        mode = self.resolve_mode(item, is_streaming, is_busy)

        if mode == QueueMode.INTERRUPT:
            # 立即中断
            callback = self._interrupt_callbacks.get(item.session_id)
            if callback:
                await callback()
            session_lane.stats.total_interrupted += 1
            # 中断后将新消息作为下一条处理
            await session_lane.enqueue(item)
            return "interrupted"

        elif mode == QueueMode.STEER:
            # 立即注入当前流
            callback = self._steer_callbacks.get(item.session_id)
            if callback and is_streaming:
                await callback(item.prompt)
                session_lane.stats.total_steered += 1
                return "steered"
            else:
                # 没有活跃流，降级为 followup
                await session_lane.enqueue(item)
                return "queued"

        elif mode == QueueMode.STEER_BACKLOG:
            # 注入 + 排队
            callback = self._steer_callbacks.get(item.session_id)
            if callback and is_streaming:
                await callback(item.prompt)
            await session_lane.enqueue(item)
            session_lane.stats.total_steered += 1
            return "steered_and_queued"

        elif mode == QueueMode.COLLECT:
            # 合并积压消息
            if is_busy and session_lane.has_pending():
                session_lane._merge_items(item)
                return "collected"
            else:
                await session_lane.enqueue(item)
                return "queued"

        elif mode == QueueMode.FOLLOWUP:
            await session_lane.enqueue(item)
            return "queued"

        else:  # QUEUE
            await session_lane.enqueue(item)
            return "queued"

    async def dequeue(self, session_id: str) -> QueueItem | None:
        """出队并获取全局执行槽位。"""
        # 先获取全局槽位
        acquired = await self.global_lane.acquire()
        if not acquired:
            return None

        session_lane = self.global_lane.get_or_create_session_lane(session_id)

        try:
            item = await asyncio.wait_for(session_lane.dequeue(), timeout=0.5)
            if item:
                session_lane._active = True
                return item
            else:
                self.global_lane.release()
                return None
        except asyncio.TimeoutError:
            self.global_lane.release()
            return None

    def mark_done(self, session_id: str) -> None:
        """标记当前任务完成。"""
        session_lane = self.global_lane.get_or_create_session_lane(session_id)
        session_lane.mark_done()
        self.global_lane.release()

    async def acquire_subagent(self) -> bool:
        """获取子 agent 执行槽位。"""
        return await self._subagent_semaphore.acquire()

    def release_subagent(self) -> None:
        self._subagent_semaphore.release()

    async def acquire_cron(self) -> bool:
        """获取 cron 执行槽位。"""
        return await self._cron_semaphore.acquire()

    def release_cron(self) -> None:
        self._cron_semaphore.release()

    def _adaptive_choice(
        self, item: QueueItem, is_streaming: bool, is_busy: bool
    ) -> QueueMode:
        """自适应模式选择。

        根据:
        - 消息优先级
        - 当前 Agent 状态
        - 系统负载

        智能选择最优处理方式。
        """
        # 高优先级 + 流式中 → steer（即时介入）
        if item.priority >= 5 and is_streaming:
            return QueueMode.STEER

        # 紧急优先级 → interrupt
        if item.priority >= 8:
            return QueueMode.INTERRUPT

        # 高负载 + 多积压 → collect（合并处理）
        if is_busy and self.global_lane.total_pending > 5:
            return QueueMode.COLLECT

        # 流式中 + 中优先级 → steer_backlog
        if is_streaming and item.priority >= 3:
            return QueueMode.STEER_BACKLOG

        # 空闲 → followup（直接处理）
        if not is_busy:
            return QueueMode.FOLLOWUP

        # 默认 → queue
        return QueueMode.QUEUE

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。"""
        return {
            "global": {
                "active": self.global_lane.stats.current_active,
                "max_concurrent": self.config.max_concurrent,
                "total_pending": self.global_lane.total_pending,
                "active_sessions": self.global_lane.active_sessions,
            },
            "sessions": {
                sid: {
                    "active": lane.is_active,
                    "queue_size": lane.stats.current_queue_size,
                    "total_enqueued": lane.stats.total_enqueued,
                    "avg_wait_ms": round(lane.stats.avg_wait_ms, 1),
                }
                for sid, lane in self.global_lane._session_lanes.items()
            },
        }
