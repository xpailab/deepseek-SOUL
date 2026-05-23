"""DeepSoul CLI — 命令行入口。

提供:
- soul chat: 交互式对话
- soul run: 单次执行
- soul gateway: 启动网关
- soul config: 配置管理
- soul status: 查看状态
- soul doctor: 诊断检查
- soul train: 启动 MLOps 训练
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soul.engine.agent import Agent
from soul.config.manager import ConfigManager

app = typer.Typer(
    name="soul",
    help="DeepSoul — 下一代 AI Agent 框架",
    add_completion=False,
)
console = Console()


# ═══════════════════════════════════════════════════════════════
# 主命令
# ═══════════════════════════════════════════════════════════════

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="直接发送的消息"),
    session: str = typer.Option("", "--session", "-s", help="会话 ID"),
    model: str = typer.Option("", "--model", "-m", help="模型名称"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="流式输出"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """启动交互式对话或发送单条消息。"""
    cfg_mgr = ConfigManager()
    config = cfg_mgr.load()

    if model:
        config.llm.model = model
    config.verbose = verbose

    agent = Agent(config=config)

    async def _run():
        await agent.initialize()

        if message:
            # 单次消息模式
            if stream:
                console.print("[bold blue]DeepSoul:[/bold blue]")
                full_response = ""
                async for chunk in agent.chat_stream(message, session_id=session):
                    if chunk.content:
                        console.print(chunk.content, end="")
                        full_response += chunk.content
                console.print()  # 换行
            else:
                with console.status("[blue]思考中..."):
                    response = await agent.chat(message, session_id=session)
                console.print(Markdown(response))
        else:
            # 交互式对话
            await _interactive_loop(agent, session)

        await agent.shutdown()

    asyncio.run(_run())


async def _interactive_loop(agent: Agent, session_id: str = ""):
    """交互式对话循环。"""
    console.print()
    console.print(Panel(
        "[bold cyan]DeepSoul Agent[/bold cyan] — 交互式对话模式\n\n"
        "输入消息开始对话，输入 / 开头的命令执行操作\n"
        "/help 查看帮助 | /status 查看状态 | /quit 退出",
        title="欢迎",
        border_style="cyan",
    ))
    console.print()

    session = await agent.sessions.get_or_create(session_id=session_id)
    session_id = session.session_id

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]再见！[/dim]")
                break
            result = await agent.handle_command(user_input, session_id)
            console.print(f"[dim]{result}[/dim]")
            continue

        # 发送消息
        console.print()
        console.print("[bold blue]DeepSoul:[/bold blue]")

        full_response = ""
        try:
            async for chunk in agent.chat_stream(user_input, session_id=session_id):
                if chunk.content:
                    console.print(chunk.content, end="")
                    full_response += chunk.content
                if chunk.tool_call:
                    console.print(
                        f"\n[dim]🔧 调用工具: {chunk.tool_call.name}[/dim]",
                        end="",
                    )
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")

        console.print("\n")


@app.command()
def run(
    task: str = typer.Argument(..., help="要执行的任务描述"),
    model: str = typer.Option("", "--model", "-m", help="模型名称"),
    output: str = typer.Option("", "--output", "-o", help="输出文件路径"),
):
    """执行单次任务。"""
    cfg_mgr = ConfigManager()
    config = cfg_mgr.load()
    if model:
        config.llm.model = model

    async def _run():
        agent = Agent(config=config)
        await agent.initialize()

        console.print(f"[bold]任务:[/bold] {task}")
        console.print(f"[bold blue]DeepSoul ({config.llm.model}):[/bold blue]")

        full = ""
        async for chunk in agent.chat_stream(task):
            if chunk.content:
                console.print(chunk.content, end="")
                full += chunk.content
        console.print()

        if output:
            Path(output).expanduser().write_text(full, encoding="utf-8")
            console.print(f"[dim]输出已保存到: {output}[/dim]")

        await agent.shutdown()

    asyncio.run(_run())


@app.command()
def config(
    key: Optional[str] = typer.Argument(None, help="配置键（如 llm.model）"),
    value: Optional[str] = typer.Argument(None, help="配置值"),
    show_all: bool = typer.Option(False, "--all", "-a", help="显示所有配置"),
    edit: bool = typer.Option(False, "--edit", "-e", help="编辑配置文件"),
):
    """管理 DeepSoul 配置。"""
    cfg_mgr = ConfigManager()

    if edit:
        cfg_path = Path.home() / ".soul" / "config.yaml"
        if not cfg_path.exists():
            cfg_mgr.save()
        console.print(f"配置文件: {cfg_path}")
        console.print("使用编辑器打开以修改配置")
        return

    if show_all or (key is None and value is None):
        config = cfg_mgr.load()
        table = Table(title="DeepSoul 配置")
        table.add_column("键", style="cyan")
        table.add_column("值", style="green")

        _flatten_config(config, table)

        console.print(table)
        return

    if key and value:
        cfg_mgr.load()
        cfg_mgr.update(**{key: value})
        cfg_mgr.save()
        console.print(f"[green]已设置 {key} = {value}[/green]")
    elif key:
        val = cfg_mgr.get(key)
        console.print(f"{key} = {val}")


def _flatten_config(config, table, prefix=""):
    """展平配置到表格。"""
    for field_name, field_info in config.model_fields.items():
        value = getattr(config, field_name)
        key = f"{prefix}.{field_name}" if prefix else field_name
        if hasattr(value, "model_fields"):
            _flatten_config(value, table, key)
        else:
            # 隐藏敏感信息
            if "api_key" in field_name and value:
                display = str(value)[:8] + "***"
            else:
                display = str(value)
            table.add_row(key, display)


@app.command()
def gateway(
    port: int = typer.Option(18789, "--port", "-p", help="网关端口"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="绑定地址"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """启动消息网关（含 REST API + WebSocket + Web 聊天界面）。"""

    async def _run():
        from soul.gateway.server import Gateway
        from soul.engine.agent import Agent
        cfg_mgr = ConfigManager()
        config = cfg_mgr.load()
        config.gateway.port = port
        config.gateway.host = host
        config.verbose = verbose

        agent = Agent(config=config)
        await agent.initialize()

        gateway = Gateway(config.gateway)
        await gateway.start(agent, host, port)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[dim]正在关闭...[/dim]")
        finally:
            await gateway.stop()

    asyncio.run(_run())


@app.command()
def status():
    """查看系统状态和统计信息。"""
    cfg_mgr = ConfigManager()
    config = cfg_mgr.load()

    console.print(Panel.fit(
        f"[bold]DeepSoul v0.1.0[/bold]\n"
        f"LLM: {config.llm.provider}/{config.llm.model}\n"
        f"工作空间: {config.memory.workspace_dir}\n"
        f"网关端口: {config.gateway.port}",
        title="系统状态",
        border_style="blue",
    ))


@app.command()
def doctor():
    """运行诊断检查。"""
    import sys

    checks = []

    # Python 版本
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python 版本", py_ver, sys.version_info >= (3, 11)))

    # 依赖检查
    deps = {
        "pydantic": "pydantic",
        "httpx": "httpx",
        "rich": "rich",
        "typer": "typer",
        "yaml": "yaml",
        "aiosqlite": "aiosqlite",
    }
    for name, module in deps.items():
        try:
            __import__(module)
            checks.append((f"依赖 {name}", "已安装", True))
        except ImportError:
            checks.append((f"依赖 {name}", "未安装", False))

    # 配置检查
    cfg_mgr = ConfigManager()
    config_path = Path.home() / ".soul" / "config.yaml"
    checks.append(("配置文件", str(config_path), config_path.exists()))

    # 工作空间检查
    ws = Path(config_path.parent / "workspace")
    checks.append(("工作空间", str(ws), ws.exists()))

    # API Key 检查
    api_key = cfg_mgr.get("llm.api_key", "")
    checks.append(("API Key 已配置", "是" if api_key else "否", bool(api_key)))

    # 输出结果
    table = Table(title="诊断检查")
    table.add_column("检查项", style="cyan")
    table.add_column("结果", style="yellow")
    table.add_column("状态")

    for name, result, ok in checks:
        status_icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(name, str(result), status_icon)

    console.print(table)

    all_ok = all(ok for _, _, ok in checks)
    if all_ok:
        console.print("\n[green]所有检查通过！[/green]")
    else:
        console.print("\n[red]部分检查未通过，请安装缺失的依赖或配置 API Key。[/red]")


@app.command()
def train(
    tasks_file: str = typer.Argument(..., help="任务列表文件 (每行一个任务)"),
    output_dir: str = typer.Option("~/.soul/training", "--output", "-o", help="输出目录"),
    workers: int = typer.Option(4, "--workers", "-w", help="并行工作数"),
    count: int = typer.Option(100, "--count", "-n", help="生成轨迹数"),
):
    """启动 MLOps 训练管道 — 批量生成训练轨迹。"""
    from soul.mlops.trajectory import TrajectoryGenerator
    from soul.types import MLOpsConfig

    tasks_path = Path(tasks_file).expanduser()
    if not tasks_path.exists():
        console.print(f"[red]文件不存在: {tasks_file}[/red]")
        raise typer.Exit(1)

    tasks = [
        line.strip()
        for line in tasks_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]

    mlops_config = MLOpsConfig(
        output_dir=output_dir,
        max_trajectories=count,
        parallel_workers=workers,
    )

    async def _train():
        cfg_mgr = ConfigManager()
        config = cfg_mgr.load()
        config.mlops = mlops_config

        agent = Agent(config=config)
        await agent.initialize()

        generator = TrajectoryGenerator(mlops_config)

        async def task_iter():
            import random
            while True:
                yield random.choice(tasks)

        console.print(f"[bold]开始生成轨迹[/bold]")
        console.print(f"  任务数: {len(tasks)}")
        console.print(f"  目标轨迹数: {count}")
        console.print(f"  工作进程: {workers}")

        trajectories = await generator.generate(task_iter(), agent, count, workers)

        console.print(f"\n[green]完成! 生成了 {len(trajectories)} 条轨迹[/green]")
        console.print(f"  输出目录: {output_dir}")

        await agent.shutdown()

    asyncio.run(_train())


@app.command()
def version():
    """显示版本信息。"""
    console.print("[bold cyan]DeepSoul v0.1.0[/bold cyan]")
    console.print("下一代 AI Agent 框架")
    console.print("融合 OpenClaw 编排能力 + Hermes 自我进化")


def main():
    """CLI 入口。"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]已取消[/dim]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
