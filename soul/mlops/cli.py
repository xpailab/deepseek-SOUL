"""MLOps CLI 入口 — 训练管道命令行。

使用:
    soul-mlops generate --tasks tasks.txt --output ./training_data
    soul-mlops compress --input ./trajectories --output ./compressed
    soul-mlops evaluate --skill skill.skill --tests tests.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="soul-mlops",
    help="DeepSoul MLOps 训练管道",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    tasks_file: str = typer.Argument(..., help="任务列表文件"),
    output_dir: str = typer.Option("~/.soul/training", "--output", "-o"),
    workers: int = typer.Option(4, "--workers", "-w"),
    count: int = typer.Option(100, "--count", "-n"),
    format: str = typer.Option("sharegpt", "--format", "-f"),
):
    """批量生成训练轨迹。"""
    from soul.mlops.trajectory import TrajectoryGenerator
    from soul.engine.agent import Agent
    from soul.config.manager import ConfigManager
    from soul.types import MLOpsConfig

    tasks_path = Path(tasks_file).expanduser()
    if not tasks_path.exists():
        console.print(f"[red]文件不存在: {tasks_file}[/red]")
        raise typer.Exit(1)

    tasks = [l.strip() for l in tasks_path.read_text(encoding="utf-8").split("\n") if l.strip()]

    mlops_config = MLOpsConfig(
        output_dir=output_dir,
        max_trajectories=count,
        parallel_workers=workers,
        output_format=format,
    )

    async def _run():
        cfg_mgr = ConfigManager()
        config = cfg_mgr.load()
        config.mlops = mlops_config
        agent = Agent(config=config)
        await agent.initialize()

        generator = TrajectoryGenerator(mlops_config)

        import random
        async def task_iter():
            while True:
                yield random.choice(tasks)

        trajs = await generator.generate(task_iter(), agent, count, workers)
        console.print(f"[green]生成了 {len(trajs)} 条轨迹[/green]")
        await agent.shutdown()

    asyncio.run(_run())


@app.command()
def compress(
    input_dir: str = typer.Argument(..., help="轨迹输入目录"),
    output_dir: str = typer.Option("", "--output", "-o", help="输出目录"),
    max_tokens: int = typer.Option(8000, "--max-tokens", "-t"),
):
    """压缩训练轨迹。"""
    from soul.mlops.compressor import TrajectoryCompressor
    from soul.types import Trajectory

    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser() if output_dir else input_path / "compressed"
    output_path.mkdir(parents=True, exist_ok=True)

    compressor = TrajectoryCompressor(max_tokens=max_tokens)

    count = 0
    for traj_file in input_path.glob("*.json"):
        try:
            data = json.loads(traj_file.read_text(encoding="utf-8"))
            traj = Trajectory(**data)
            compressed = compressor.compress(traj)
            out_file = output_path / traj_file.name
            out_file.write_text(compressed.model_dump_json(indent=2), encoding="utf-8")
            count += 1
        except Exception as e:
            console.print(f"[yellow]跳过 {traj_file.name}: {e}[/yellow]")

    console.print(f"[green]压缩了 {count} 条轨迹 → {output_path}[/green]")


@app.command()
def evaluate(
    skill_file: str = typer.Argument(..., help="技能文件路径"),
    tests_file: str = typer.Option("", "--tests", "-t", help="测试用例 JSON"),
):
    """评估技能质量。"""
    from soul.mlops.evaluator import LLMJudge

    skill_path = Path(skill_file).expanduser()
    if not skill_path.exists():
        console.print(f"[red]文件不存在: {skill_file}[/red]")
        raise typer.Exit(1)

    skill_content = skill_path.read_text(encoding="utf-8")
    judge = LLMJudge()
    result = judge._rule_evaluate_skill(skill_content)

    table = Table(title=f"技能评估: {skill_path.name}")
    table.add_column("维度", style="cyan")
    table.add_column("分数", style="green")
    for k, v in result.items():
        table.add_row(k, str(v))

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
