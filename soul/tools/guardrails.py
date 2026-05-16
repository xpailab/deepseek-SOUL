"""工具安全护栏 — 调用前校验、路径沙箱、命令审批。

四层防护：
1. 参数校验 — 检查参数合法性和注入攻击
2. 路径沙箱 — 限制文件操作范围
3. 命令审批 — 高风险命令需用户确认
4. Shell 钩子 — 拦截危险系统调用
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from soul.types import ToolRisk


class ToolGuardrails:
    """工具安全护栏。

    在工具调用前执行安全检查，阻止危险操作。
    """

    # 危险命令模式
    DANGEROUS_COMMANDS: list[str] = [
        r"rm\s+(-rf?|--recursive).*\/",    # rm -rf /
        r"mkfs\.",                          # 格式化
        r"dd\s+if=",                        # dd
        r">\s*/dev/sd",                     # 覆盖磁盘
        r"chmod\s+777\s+/",                 # 过度权限
        r"chown\s+-R\s+.*\/",              # 递归 chown
        r":\(\)\s*\{\s*:\s*\|\:",           # fork bomb
        r"wget.*\|.*sh",                    # curl pipe sh
        r"curl.*\|.*bash",                  # curl pipe bash
        r"eval\s+.*\$",                     # eval 变量
        r"sudo\s+.*rm\s+-rf",              # sudo rm
        r"docker\s+rm\s+-f.*all",          # 删除所有容器
        r"kubectl\s+delete\s+all",          # 删除所有 k8s 资源
    ]

    # Prompt injection 检测模式
    INJECTION_PATTERNS: list[str] = [
        r"<system.reminder>",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"忽略.*指令",
        r"ignore.*instructions",
        r"you are now",
        r"新的.*角色",
        r"扮演",
        r"DAN\s",
        r"jailbreak",
    ]

    def __init__(
        self,
        workspace_dir: str = "~/.soul/workspace",
        allowed_paths: list[str] | None = None,
        blocked_paths: list[str] | None = None,
    ):
        self.workspace = Path(workspace_dir).expanduser().resolve()
        self.allowed_paths: list[Path] = [
            self.workspace,
            *(Path(p).expanduser().resolve() for p in (allowed_paths or [])),
        ]
        self.blocked_paths: list[Path] = [
            Path(p).expanduser().resolve() for p in (blocked_paths or [])
        ] or [
            Path("/etc/passwd"),
            Path("/etc/shadow"),
        ]
        self._dangerous_re = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_COMMANDS]
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        risk: ToolRisk = ToolRisk.SAFE,
    ) -> tuple[bool, str]:
        """检查工具调用是否安全。

        Returns:
            (is_safe, reason)
        """
        # 检查参数注入
        args_str = str(arguments).lower()
        for pattern in self._injection_re:
            if pattern.search(args_str):
                return False, f"检测到 prompt injection: {pattern.pattern}"

        # 按工具名检查
        if tool_name in ("bash", "shell", "exec", "run"):
            return self._check_command(arguments.get("command", ""))
        elif tool_name in ("write", "write_file", "edit", "edit_file"):
            return self._check_file_path(arguments.get("file_path", ""))
        elif tool_name in ("read", "read_file"):
            return self._check_read_path(arguments.get("file_path", ""))
        elif tool_name in ("delete", "delete_file", "rm"):
            return self._check_delete_path(arguments.get("file_path", ""))

        return True, "OK"

    def check_command(self, command: str) -> tuple[bool, str]:
        """检查 shell 命令是否安全。"""
        return self._check_command(command)

    def check_file_access(self, filepath: str, write: bool = False) -> tuple[bool, str]:
        """检查文件访问权限。"""
        try:
            path = Path(filepath).expanduser().resolve()
        except Exception:
            return False, f"无效路径: {filepath}"

        # 检查是否在阻止列表中
        for blocked in self.blocked_paths:
            try:
                path.relative_to(blocked)
                return False, f"路径被阻止: {filepath}"
            except ValueError:
                pass

        # 写入需要检查是否在允许列表中
        if write:
            for allowed in self.allowed_paths:
                try:
                    path.relative_to(allowed)
                    return True, "OK"
                except ValueError:
                    pass
            return False, f"写入路径不在允许范围内: {filepath}"

        return True, "OK"

    def scan_prompt_injection(self, text: str) -> tuple[bool, list[str]]:
        """扫描文本中的 prompt injection 攻击。

        Returns:
            (is_safe, found_patterns)
        """
        found: list[str] = []
        for pattern in self._injection_re:
            if pattern.search(text):
                found.append(pattern.pattern)
        return len(found) == 0, found

    def _check_command(self, command: str) -> tuple[bool, str]:
        """检查命令安全性。"""
        if not command.strip():
            return True, "OK"

        for pattern in self._dangerous_re:
            if pattern.search(command):
                return False, f"危险命令被拦截: 匹配模式 '{pattern.pattern}'"

        # 检查是否在允许的路径中操作
        path_refs = re.findall(r'["\']?(\/[^\s"\']+)["\']?', command)
        for path_str in path_refs:
            is_safe, reason = self.check_file_access(path_str, write=True)
            if not is_safe:
                return False, reason

        return True, "OK"

    def _check_file_path(self, filepath: str) -> tuple[bool, str]:
        return self.check_file_access(filepath, write=True)

    def _check_read_path(self, filepath: str) -> tuple[bool, str]:
        return self.check_file_access(filepath, write=False)

    def _check_delete_path(self, filepath: str) -> tuple[bool, str]:
        if not filepath:
            return False, "删除路径为空"
        return self.check_file_access(filepath, write=True)

    def get_approval_prompt(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """生成用户审批提示。"""
        return (
            f"⚠️ 高风险操作需要确认\n"
            f"工具: {tool_name}\n"
            f"参数: {arguments}\n"
            f"请输入 'yes' 确认执行，或 'no' 取消:"
        )
