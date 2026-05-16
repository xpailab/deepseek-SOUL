"""批量轨迹生成器。

并行生成数千条工具调用轨迹，用于模型微调和 GEPA 评估。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from soul.types import MLOpsConfig, Trajectory, TrajectoryStep


class TrajectoryGenerator:
    """批量轨迹生成器。

    功能:
    - 并行生成多条工具调用轨迹
    - 自动检查点（中断可续）
    - 可配置工作进程数和工具分布
    - 输出格式: ShareGPT, OpenAI, Claude
    """

    def __init__(self, config: MLOpsConfig | None = None):
        self.config = config or MLOpsConfig()
        self.output_dir = Path(self.config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_file = self.output_dir / "checkpoint.json"
        self._generated: int = 0
        self._errors: int = 0

    async def generate(
        self,
        task_generator: Any,  # AsyncIterator[str] — 任务描述生成器
        agent: Any,  # Agent 实例
        num_trajectories: int | None = None,
        parallel_workers: int | None = None,
    ) -> list[Trajectory]:
        """并行生成轨迹。

        Args:
            task_generator: 产生任务描述的异步迭代器
            agent: Agent 实例，用于执行任务
            num_trajectories: 生成轨迹总数
            parallel_workers: 并行工作进程数

        Returns:
            生成的轨迹列表
        """
        num = num_trajectories or self.config.max_trajectories
        workers = parallel_workers or self.config.parallel_workers

        # 恢复检查点
        generated_trajs: list[Trajectory] = self._load_checkpoint()
        start_index = len(generated_trajs)

        if start_index > 0:
            print(f"从检查点恢复: 已有 {start_index} 条轨迹")

        # 生产者-消费者模式
        task_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=workers * 2)

        async def producer():
            count = start_index
            async for task in task_generator:
                if count >= num:
                    break
                await task_queue.put(task)
                count += 1
            # 发送结束信号
            for _ in range(workers):
                await task_queue.put(None)

        async def worker(worker_id: int):
            while True:
                task = await task_queue.get()
                if task is None:
                    task_queue.task_done()
                    break

                try:
                    traj = await self._execute_task(agent, task)
                    if traj:
                        generated_trajs.append(traj)
                        self._generated += 1
                        # 定期保存检查点
                        if self._generated % 10 == 0:
                            self._save_checkpoint(generated_trajs)
                except Exception as e:
                    self._errors += 1
                finally:
                    task_queue.task_done()

        # 启动生产者和消费者
        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker(i)) for i in range(workers)]

        await producer_task
        await task_queue.join()
        await asyncio.gather(*worker_tasks)

        # 最终保存
        self._save_checkpoint(generated_trajs)
        self._export_trajectories(generated_trajs)

        return generated_trajs

    async def _execute_task(self, agent: Any, task: str) -> Trajectory | None:
        """执行单个任务并记录轨迹。"""
        start = time.time()
        steps: list[TrajectoryStep] = []
        session_id = ""

        try:
            # 使用 Agent 执行任务
            result = await agent.chat(task)

            # 从 Agent 会话中提取步骤
            session = await agent.sessions.get(session_id) if session_id else None

            # 构建轨迹
            step = TrajectoryStep(
                step_index=0,
                role="user",
                content=task,
            )
            steps.append(step)

            step = TrajectoryStep(
                step_index=1,
                role="assistant",
                content=result,
                duration_ms=(time.time() - start) * 1000,
            )
            steps.append(step)

            return Trajectory(
                session_id=session_id or "batch",
                task=task,
                steps=steps,
                success=True,
                total_duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return Trajectory(
                session_id=session_id or "batch",
                task=task,
                steps=steps,
                success=False,
                metadata={"error": str(e)},
            )

    def _save_checkpoint(self, trajectories: list[Trajectory]) -> None:
        """保存检查点。"""
        if not self.config.checkpoint_enabled:
            return

        data = {
            "count": len(trajectories),
            "timestamp": time.time(),
            "ids": [t.id for t in trajectories],
        }
        self._checkpoint_file.write_text(json.dumps(data), encoding="utf-8")

    def _load_checkpoint(self) -> list[Trajectory]:
        """加载检查点。"""
        if not self._checkpoint_file.exists():
            return []

        try:
            data = json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
            # 尝试加载轨迹文件
            trajs: list[Trajectory] = []
            for traj_file in sorted(self.output_dir.glob("traj_*.json")):
                try:
                    traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
                    trajs.append(Trajectory(**traj_data))
                except Exception:
                    pass
            return trajs
        except Exception:
            return []

    def _export_trajectories(self, trajectories: list[Trajectory]) -> None:
        """导出轨迹到指定格式。"""
        fmt = self.config.output_format

        if fmt == "sharegpt":
            self._export_sharegpt(trajectories)
        elif fmt == "openai":
            self._export_openai(trajectories)
        elif fmt == "claude":
            self._export_claude(trajectories)

        # 同时保存原始格式
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for traj in trajectories:
            filepath = raw_dir / f"{traj.id}.json"
            filepath.write_text(
                traj.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def _export_sharegpt(self, trajectories: list[Trajectory]) -> None:
        """导出 ShareGPT 格式。"""
        data = []
        for traj in trajectories:
            conversations = []
            for step in traj.steps:
                if step.role.value == "system":
                    conversations.append({"from": "system", "value": step.content})
                elif step.role.value == "user":
                    conversations.append({"from": "human", "value": step.content})
                elif step.role.value == "assistant":
                    conversations.append({"from": "gpt", "value": step.content})

            data.append({
                "id": traj.id,
                "conversations": conversations,
            })

        filepath = self.output_dir / "sharegpt_trajectories.json"
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _export_openai(self, trajectories: list[Trajectory]) -> None:
        """导出 OpenAI 微调格式（JSONL）。"""
        filepath = self.output_dir / "openai_trajectories.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for traj in trajectories:
                messages = []
                for step in traj.steps:
                    messages.append({
                        "role": step.role.value,
                        "content": step.content,
                    })
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    def _export_claude(self, trajectories: list[Trajectory]) -> None:
        """导出 Claude 格式。"""
        data = []
        for traj in trajectories:
            turns = []
            for step in traj.steps:
                if step.role.value not in ("system",):
                    turns.append({
                        "role": step.role.value,
                        "content": step.content,
                    })
            data.append({
                "id": traj.id,
                "turns": turns,
            })

        filepath = self.output_dir / "claude_trajectories.json"
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_stats(self) -> dict[str, Any]:
        return {
            "generated": self._generated,
            "errors": self._errors,
            "output_dir": str(self.output_dir),
            "format": self.config.output_format,
        }
