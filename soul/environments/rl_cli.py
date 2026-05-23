"""RL 训练 CLI — 轨迹生成 → 奖励标注 → 训练数据导出。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(name="soul-rl", help="Atropos RL 训练工具")


@app.command("parse")
def parse_tool_calls(
    input_file: str = typer.Argument(..., help="LLM 输出文本文件"),
    output: str = typer.Option("", help="输出 JSON 文件路径"),
) -> None:
    """从 LLM 输出中解析工具调用。"""
    from soul.environments.atropos import AtroposEnv

    text = Path(input_file).read_text(encoding="utf-8")
    calls = AtroposEnv.parse_tool_calls(text)

    result = {
        "source": input_file,
        "text_length": len(text),
        "tool_calls": calls,
        "tool_chain": [c["name"] for c in calls],
    }

    if output:
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"解析完成: {len(calls)} 个工具调用 → {output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("reward")
def generate_rewards(
    trajectory_file: str = typer.Argument(..., help="轨迹 JSON 文件"),
    output: str = typer.Option("rewards.jsonl", help="输出 JSONL 文件"),
) -> None:
    """为轨迹生成 RL 奖励标注。"""
    from soul.environments.atropos import AtroposEnv

    data = json.loads(Path(trajectory_file).read_text(encoding="utf-8"))
    trajectories = data if isinstance(data, list) else [data]

    with open(output, "w", encoding="utf-8") as f:
        for traj in trajectories:
            rl_data = AtroposEnv.trajectory_to_rl_data(traj)
            f.write(json.dumps(rl_data, ensure_ascii=False) + "\n")

    print(f"奖励标注完成: {len(trajectories)} 条轨迹 → {output}")


@app.command("export")
def export_training_data(
    trajectories_dir: str = typer.Argument(..., help="轨迹目录"),
    output: str = typer.Option("training_data.jsonl", help="输出文件"),
    format: str = typer.Option("chatml", help="输出格式: chatml / openai / sharegpt"),
) -> None:
    """导出 RL 训练数据。"""
    from soul.environments.atropos import AtroposEnv

    traj_dir = Path(trajectories_dir)
    count = 0

    with open(output, "w", encoding="utf-8") as outf:
        for f in traj_dir.glob("*.json"):
            try:
                traj = json.loads(f.read_text(encoding="utf-8"))
                rl_data = AtroposEnv.trajectory_to_rl_data(traj, format=format)
                outf.write(json.dumps(rl_data, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                pass

    print(f"导出完成: {count} 条训练数据 → {output}")


@app.command("simulate")
def simulate(
    task: str = typer.Argument(..., help="模拟任务描述"),
    steps: int = typer.Option(5, help="最大步数"),
) -> None:
    """模拟一次 RL episode（不调用真实 LLM）。"""
    from soul.environments.atropos import AtroposEnv, AtroposConfig

    config = AtroposConfig(max_steps=steps)
    env = AtroposEnv(config)

    tools = [
        {"name": "bash", "description": "执行 shell 命令"},
        {"name": "file", "description": "读写文件"},
        {"name": "web", "description": "HTTP 请求"},
    ]

    state = env.reset(task=task, tools=tools)
    print(f"状态: {state['task']}")
    print(f"可用工具: {[t['name'] for t in tools]}")
    print(f"最大步数: {steps}")
    print("---")
    print("模拟工具调用序列...")
    print("（实际使用时会由 LLM Agent 选择每个 step 的 action）")


if __name__ == "__main__":
    app()
