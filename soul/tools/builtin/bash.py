"""Bash 执行工具 — 在沙箱中运行 shell 命令。"""

import asyncio
import locale
import os
from pathlib import Path
from typing import Any

from soul.tools.registry import ToolDef
from soul.types import ToolRisk


def _decode_output(data: bytes) -> str:
    """智能解码命令输出。Windows 上先试 GBK，后试 UTF-8。"""
    sys_enc = locale.getpreferredencoding() or "gbk"
    # 按优先级尝试：系统编码 → UTF-8
    for enc in [sys_enc, "utf-8", "gbk", "latin-1"]:
        try:
            text = data.decode(enc)
            # 如果替换字符占比 < 5%，认为解码正确
            if len(text) > 0 and text.count("\ufffd") / len(text) < 0.05:
                return text
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


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

    # PowerShell 命令特征
    _PS_CMDS = {"Get-", "Set-", "New-", "Remove-", "Start-", "Stop-", "Invoke-",
                "Write-", "Out-", "Select-", "Where-", "ForEach-", "Test-",
                "Format-", "Export-", "Import-", "ConvertTo-", "ConvertFrom-",
                "Copy-Item", "Move-Item", "Rename-Item", "ls", "dir", "cat",
                "mkdir", "rm", "cp", "mv", "pwd", "echo", "type", "cd"}

    def _needs_powershell(self, command: str) -> bool:
        """检查命令是否需要 PowerShell 执行。"""
        # 明确包含 PS cmdlet 的特征
        ps_patterns = [
            "Get-", "Set-", "New-", "Remove-", "Start-", "Stop-", "Invoke-",
            "Write-", "Out-", "Select-Object", "Where-Object", "ForEach-Object",
            "Test-", "Format-", "Export-", "Import-", "ConvertTo-", "ConvertFrom-",
            "Copy-Item", "Move-Item", "Rename-Item",
        ]
        cmd_lower = command.lower()
        for p in ps_patterns:
            if p.lower() in cmd_lower:
                return True
        return False

    def _is_windows(self) -> bool:
        import platform
        return platform.system() == "Windows"

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
            proc_env = {**os.environ, "SOUL_WORKSPACE": str(self.workspace)}

            # Windows 上检测是否需要 PowerShell vs cmd
            if self._is_windows():
                if self._needs_powershell(command):
                    # PowerShell 命令 → 包装为 powershell -Command
                    escaped = command.replace("'", "''")
                    wrapped = f'powershell -NoProfile -Command "{escaped}"'
                else:
                    # 普通 cmd 命令
                    wrapped = f'cmd /c "{command}"'
            else:
                wrapped = command

            proc = await asyncio.create_subprocess_shell(
                wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=proc_env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            result = {
                "exit_code": proc.returncode or 0,
                "stdout": _decode_output(stdout)[:10000],
                "stderr": _decode_output(stderr)[:5000],
                "cwd": cwd,
            }

            # 如果 cmd 执行失败且看起来像 PS 命令，自动重试用 PowerShell
            if self._is_windows() and proc.returncode != 0 and not self._needs_powershell(command):
                ps_keywords = ["Get-ChildItem", "ls", "dir", "cat", "mkdir", "rm", "cp", "mv", "echo", "type", "Select-", "Where-", "ForEach-"]
                if any(kw in command for kw in ps_keywords):
                    escaped2 = command.replace("'", "''")
                    wrapped2 = f'powershell -NoProfile -Command "{escaped2}"'
                    proc2 = await asyncio.create_subprocess_shell(
                        wrapped2,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd,
                        env=proc_env,
                    )
                    stdout2, stderr2 = await asyncio.wait_for(
                        proc2.communicate(), timeout=timeout
                    )
                    return {
                        "exit_code": proc2.returncode or 0,
                        "stdout": _decode_output(stdout2)[:10000],
                        "stderr": _decode_output(stderr2)[:5000],
                        "cwd": cwd,
                        "retried_with": "powershell",
                    }

            return result

        except TimeoutError:
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
