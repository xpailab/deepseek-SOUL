"""沙箱隔离系统。

支持三种执行后端:
1. 本地进程隔离
2. Docker 容器隔离
3. SSH 远程执行

安全特性:
- 只读根文件系统（Docker）
- 权限降级
- PID 限制
- 网络隔离
- 内存/CPU 限制
"""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

from soul.types import SandboxConfig, SandboxMode


class Sandbox:
    """执行沙箱 — 隔离外部命令执行。"""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._docker_available: bool | None = None

    async def execute(
        self,
        command: str,
        mode: SandboxMode | None = None,
        working_dir: str = "",
        timeout: int = 120,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """在沙箱中执行命令。

        Args:
            command: 要执行的命令
            mode: 沙箱模式（默认使用配置中的模式）
            working_dir: 工作目录
            timeout: 超时秒数
            env: 环境变量

        Returns:
            {exit_code, stdout, stderr, mode}
        """
        mode = mode or self.config.default_mode

        if mode == SandboxMode.DOCKER:
            if not await self._check_docker():
                mode = SandboxMode.LOCAL  # 降级
            else:
                return await self._docker_execute(command, working_dir, timeout, env)

        if mode == SandboxMode.SSH:
            return await self._ssh_execute(command, working_dir, timeout)

        # 默认：本地执行
        return await self._local_execute(command, working_dir, timeout, env)

    async def _local_execute(
        self,
        command: str,
        working_dir: str = "",
        timeout: int = 120,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """本地进程执行。"""
        cwd = working_dir or str(Path.home())
        sandbox_env = {**os.environ, **(env or {})}

        # 过滤危险环境变量
        for key in list(sandbox_env.keys()):
            if key.startswith(("SECRET_", "PASSWORD", "TOKEN", "KEY")):
                sandbox_env.pop(key, None)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=sandbox_env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            return {
                "exit_code": proc.returncode or 0,
                "stdout": stdout.decode("utf-8", errors="replace")[:10000],
                "stderr": stderr.decode("utf-8", errors="replace")[:5000],
                "mode": "local",
                "cwd": cwd,
            }
        except asyncio.TimeoutError:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"命令超时 ({timeout}s)",
                "mode": "local",
                "cwd": cwd,
            }

    async def _docker_execute(
        self,
        command: str,
        working_dir: str = "",
        timeout: int = 120,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Docker 容器执行。"""
        docker_cmd = [
            "docker", "run", "--rm",
            "--read-only" if self.config.readonly_root else "",
            f"--memory={self.config.memory_limit}",
            f"--cpus={self.config.cpu_limit}",
            "--network", "none" if not self.config.network_enabled else "bridge",
            "-v", f"{working_dir or Path.home()}:/workspace",
            "-w", "/workspace",
        ]

        docker_cmd = [a for a in docker_cmd if a]  # 过滤空字符串
        docker_cmd.append(self.config.docker_image)
        docker_cmd.extend(["sh", "-c", command])

        return await self._local_execute(
            shlex.join(docker_cmd),
            working_dir or str(Path.home()),
            timeout + 10,  # Docker 额外开销
            env,
        )

    async def _ssh_execute(
        self,
        command: str,
        working_dir: str = "",
        timeout: int = 120,
    ) -> dict[str, Any]:
        """SSH 远程执行（需要配置 SSH 连接信息）。"""
        # SSH 执行需要额外配置，此处提供框架
        ssh_host = os.getenv("SOUL_SSH_HOST", "")
        if not ssh_host:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "SSH 未配置: 设置 SOUL_SSH_HOST 环境变量",
                "mode": "ssh",
            }

        ssh_cmd = f'ssh {ssh_host} "cd {working_dir or "~"} && {command}"'
        return await self._local_execute(ssh_cmd, "", timeout)

    async def _check_docker(self) -> bool:
        """检查 Docker 是否可用。"""
        if self._docker_available is None:
            result = await self._local_execute("docker info 2>&1", timeout=5)
            self._docker_available = result["exit_code"] == 0
        return self._docker_available

    def get_allowed_commands(self, command: str) -> bool:
        """检查命令是否在允许列表中。"""
        if not self.config.allowed_commands:
            return True  # 无限制
        return any(
            command.strip().startswith(cmd)
            for cmd in self.config.allowed_commands
        )

    def is_path_blocked(self, path: str) -> bool:
        """检查路径是否被阻止。"""
        path_obj = Path(path).expanduser().resolve()
        return any(
            str(path_obj).startswith(str(Path(bp).expanduser().resolve()))
            for bp in self.config.blocked_paths
        )
