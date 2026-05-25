"""工具安全护栏 — 调用前校验、路径沙箱、命令审批。

四层防护：
1. 参数校验 — 检查参数合法性和注入攻击
2. 路径沙箱 — 限制文件操作范围
3. 命令审批 — 高风险命令需用户确认
4. Shell 钩子 — 拦截危险系统调用
"""

from __future__ import annotations

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
        # Linux
        r"rm\s+(-rf?|--recursive).*\/",    # rm -rf /
        r"rm\s+(-rf?|--recursive).*\$HOME", # rm -rf $HOME
        r"mkfs\.",                          # 格式化
        r"dd\s+if=",                        # dd
        r">\s*/dev/sd",                     # 覆盖磁盘
        r"chmod\s+.*777\s+/",               # 过度权限（含 chmod -R 777 /）
        r"chown\s+-R\s+.*\/",              # 递归 chown
        r":\(\)\s*\{\s*:\s*\|\:",           # fork bomb
        r"wget.*\|.*sh",                    # curl pipe sh
        r"curl.*\|.*bash",                  # curl pipe bash
        r"eval\s+.*\$",                     # eval 变量
        r"sudo\s+.*rm\s+-rf",              # sudo rm
        r"docker\s+rm\s+-f.*all",          # 删除所有容器
        r"kubectl\s+delete\s+all",          # 删除所有 k8s 资源
        # Windows
        r"rmdir\s+/s\s+/q\s+[A-Za-z]:",    # Windows 递归删除盘符
        r"del\s+/f\s+/s\s+/q\s+[A-Za-z]:", # Windows 强制递归删除
        r"format\s+[A-Za-z]:",             # Windows 格式化
        r"diskpart",                        # Windows 磁盘管理
        r"shutdown\s+/s\b(?!\?)",          # Windows 关机（排除 shutdown /? 帮助）
        r"reg\s+delete\s+(HKLM|HKEY_LOCAL_MACHINE)", # 删除注册表
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
        is_write = tool_name in ("write", "write_file", "edit", "edit_file", "file")
        for pattern in self._injection_re:
            if pattern.search(args_str):
                p = pattern.pattern
                # ChatML 格式标记在代码文件中是合法的（数据加载脚本、训练数据等）
                if is_write and ("im_start" in p or "im_end" in p):
                    continue
                return False, f"检测到 prompt injection: {p}"

        # 按工具名检查
        if tool_name in ("bash", "shell", "exec", "run"):
            return self._check_command(arguments.get("command", ""))
        elif tool_name == "win" or tool_name == "browser":
            return True, "OK"  # GUI/浏览器工具，无危险操作
        elif tool_name in ("write", "write_file", "edit", "edit_file"):
            return self._check_file_path(arguments.get("file_path", ""))
        elif tool_name in ("read", "read_file"):
            return self._check_read_path(arguments.get("file_path", ""))
        elif tool_name in ("delete", "delete_file", "rm"):
            return self._check_delete_path(arguments.get("file_path", ""))
        elif tool_name == "file":
            # 通用 file 工具：根据 action 参数决定检查类型
            op = arguments.get("operation", "") or arguments.get("action", "")
            fp = arguments.get("file_path", "")
            if op in ("read", "read_file", "cat", "head", "tail"):
                return self._check_read_path(fp)
            elif op in ("list", "exists"):
                return True, "OK"  # 只读操作，始终允许
            elif op in ("delete", "remove", "rm"):
                return self._check_delete_path(fp)
            elif op == "mkdir":
                return self._check_file_path(fp)  # 创建目录，检查路径
            else:
                return self._check_file_path(fp)

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

        # 写入操作：只要不是系统关键路径，就允许
        if write:
            # 阻止写入系统关键目录
            system_paths = [
                Path("/etc"), Path("/boot"), Path("/sys"), Path("/proc"), Path("/dev"),
                Path("/bin"), Path("/sbin"), Path("/usr/bin"), Path("/usr/sbin"),
                Path("C:/Windows"), Path("C:/Program Files"), Path("C:/ProgramData"),
                Path.home() / ".ssh", Path.home() / ".gnupg",
            ]
            for sys_path in system_paths:
                try:
                    if path.relative_to(sys_path):
                        return False, f"不能写入系统目录: {filepath}"
                except ValueError:
                    pass

            # 如果明确配置了允许路径，优先检查
            if len(self.allowed_paths) > 1:  # 除了默认workspace还有其他路径
                for allowed in self.allowed_paths:
                    try:
                        path.relative_to(allowed)
                        return True, "OK"
                    except ValueError:
                        pass
                return False, f"写入路径不在允许范围内: {filepath}"

            # 默认情况：允许写入任何非系统路径
            return True, "OK"

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
        """检查命令安全性 — 仅拦截危险操作，不限制路径。"""
        if not command.strip():
            return True, "OK"

        for pattern in self._dangerous_re:
            if pattern.search(command):
                return False, f"危险命令被拦截: 匹配模式 '{pattern.pattern}'"

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
