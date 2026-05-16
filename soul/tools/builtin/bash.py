"""Bash 执行工具 — 在沙箱中运行 shell 命令。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from soul.tools.registry import ToolDef
from soul.types import ToolRisk


class BashTool:
    """Shell 命令执行工具。"""

    NAME = "bash"
    DESCRIPTION = "在沙箱终端中执行 shell 命令。返回 stdout 和 stderr。"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令",
            },
            "working_dir": {
                "type": "string",
                "description": "工作目录（可选）",
            },
            "timeout": {
                "type": "integer",
                "description": "超时秒数，默认 120",
                "default": 120,
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()

    async def execute(
        self,
        command: str,
        working_dir: str = "",
        timeout: int = 120,
    ) -> dict[str, Any]:
        """执行 shell 命令。"""
        cwd = str(self.workspace)
        if working_dir:
            cwd = str(Path(working_dir).expanduser().resolve())

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ, "SOUL_WORKSPACE": str(self.workspace)},
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            return {
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[:10000],
                "stderr": stderr.decode("utf-8", errors="replace")[:5000],
                "cwd": cwd,
            }

        except asyncio.TimeoutError:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"命令超时 ({timeout}s)",
                "cwd": cwd,
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "cwd": cwd,
            }

    @classmethod
    def to_tool_def(cls) -> ToolDef:
        return ToolDef(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            handler=cls().execute,
            parameters=cls.PARAMETERS,
            risk=ToolRisk.HIGH,
            requires_approval=False,
            timeout_seconds=120,
            max_retries=0,
            tags=["shell", "exec", "system"],
        )
