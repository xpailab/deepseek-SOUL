"""结果验证器 — 工具执行后验证输出是否符合预期。

三层验证:
1. 结构化验证（快速，无 LLM）：文件存在/非空/语法/错误关键词
2. 模式匹配验证（快速）：预期输出模式是否匹配
3. 语义验证（下一轮 LLM）：通过工作记忆注入验证上下文

验证失败时自动生成修正建议，集成到自纠错闭环。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerifyResult:
    """验证结果。"""
    passed: bool
    tool_name: str = ""
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    severity: str = "info"  # info / warning / error

    def to_prompt_context(self) -> str:
        """生成可注入 prompt 的验证上下文。"""
        if self.passed:
            return ""
        lines = [f"\n[验证失败] {self.tool_name} 输出不符合预期:"]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        if self.suggestions:
            lines.append("  修正建议:")
            for s in self.suggestions:
                lines.append(f"  → {s}")
        return "\n".join(lines)


class ResultVerifier:
    """工具结果验证器。

    在每个工具执行后快速检查输出质量，
    发现问题时生成修正建议，反馈到工作记忆。
    """

    # 危险信号——stdout 中的错误关键词，即使 exit_code=0
    ERROR_SIGNALS: list[tuple[str, str]] = [
        (r"error", "输出中包含 'error'"),
        (r"Traceback\s*\(most recent call last\)", "Python Traceback 错误"),
        (r"fatal:", "Git/Docker 致命错误"),
        (r"cannot\s+find", "找不到目标文件或命令"),
        (r"permission\s+denied", "权限不足"),
        (r"command\s+not\s+found", "命令不存在"),
        (r"ModuleNotFoundError", "Python 模块未安装"),
        (r"SyntaxError", "Python 语法错误"),
        (r"connection\s+refused", "连接被拒绝"),
        (r"EACCES", "文件权限错误 (EACCES)"),
        (r"ENOENT", "文件不存在 (ENOENT)"),
        (r"cannot\s+stat", "文件路径无效"),
        (r"no\s+such\s+file", "文件不存在"),
        (r"npm\s+ERR!", "npm 错误"),
        (r"pip\s+ERROR", "pip 安装错误"),
        (r"cargo\s+error", "Rust 编译错误"),
        (r"go\s+:\s+.*not\s+found", "Go 依赖未找到"),
    ]

    # 成功信号——确认输出有效
    SUCCESS_SIGNALS: list[tuple[str, str]] = [
        (r"successfully", "明确成功"),
        (r"completed", "任务完成"),
        (r"installed", "安装成功"),
        (r"created", "创建成功"),
        (r"ok", "OK 状态"),
        (r"running", "服务运行中"),
        (r"active", "服务活跃"),
    ]

    def verify_tool_result(
        self,
        tool_name: str,
        result: Any,
        error: str = "",
        exit_code: int | None = None,
        expected: str = "",
        context: dict[str, Any] | None = None,
    ) -> VerifyResult:
        """验证工具执行结果。

        Args:
            tool_name: 工具名称
            result: 工具返回的结果（字符串或字典）
            error: 错误信息
            exit_code: 进程退出码（bash 工具）
            expected: 计划中的预期结果描述
            context: 额外上下文

        Returns:
            VerifyResult: 验证结果，含问题和修正建议
        """
        vr = VerifyResult(passed=True, tool_name=tool_name)

        # 1. 结果为空检查
        if self._is_empty(result) and not error:
            vr.passed = False
            vr.issues.append("工具返回空结果——命令可能未执行或输出被截断")
            vr.suggestions.append("重新执行并检查命令是否正确")
            vr.severity = "error"

        # 2. 按工具类型进行专项检查
        if tool_name in ("bash", "shell", "exec", "run"):
            self._verify_shell(result, error, exit_code, expected, vr)
        elif tool_name in ("write_file", "write", "edit", "edit_file", "file"):
            self._verify_file_write(tool_name, result, context, vr)
        elif tool_name in ("read_file", "read", "cat"):
            self._verify_file_read(result, vr)
        elif tool_name == "web":
            self._verify_web(result, vr)

        return vr

    def verify_plan_step(
        self, step_action: str, result: Any, expected: str
    ) -> VerifyResult:
        """对照计划步骤验证——检查结果是否满足计划的预期。"""
        vr = VerifyResult(passed=True, tool_name="plan_step")
        if not expected:
            return vr

        result_str = str(result).lower() if result else ""

        # 检查预期关键词是否在结果中
        keywords = re.findall(r'[一-鿿\w]+', expected.lower())
        matched = [kw for kw in keywords if len(kw) >= 2 and kw in result_str]
        missed = [kw for kw in keywords if len(kw) >= 2 and kw not in result_str]

        if len(missed) > len(matched) and len(missed) >= 2:
            vr.passed = False
            vr.issues.append(f"预期结果未达到: {expected}")
            vr.issues.append(f"输出中缺少: {', '.join(missed[:5])}")
            vr.suggestions.append(f"检查 {step_action} 是否真正执行成功")
            vr.severity = "warning"

        return vr

    # ═══ 专项验证 ═══

    def _verify_shell(self, result, error, exit_code, expected, vr):
        """验证 shell 命令输出。"""
        output = self._to_str(result)

        # 错误输出
        if error:
            vr.passed = False
            vr.issues.append(f"stderr: {str(error)[:200]}")
            vr.suggestions.append("检查命令参数是否正确")
            vr.severity = "error"
            # 分析常见错误
            suggestion = self._analyze_shell_error(str(error))
            if suggestion:
                vr.suggestions.append(suggestion)
            return

        # exit_code 非 0
        if exit_code is not None and exit_code != 0:
            vr.passed = False
            vr.issues.append(f"进程退出码: {exit_code}")
            vr.suggestions.append("检查命令是否执行成功")
            vr.severity = "error"
            return

        # 输出中隐藏的错误信号
        for pattern, description in self.ERROR_SIGNALS:
            if re.search(pattern, output, re.IGNORECASE):
                vr.passed = False
                vr.issues.append(description)
                vr.severity = "warning"

        if not vr.passed:
            suggestion = self._analyze_shell_error(output)
            if suggestion:
                vr.suggestions.append(suggestion)

    def _verify_file_write(self, tool_name, result, context, vr):
        """验证文件写入。"""
        # 尝试从 context 或 result 中获取路径
        filepath = ""
        if context:
            filepath = context.get("file_path", context.get("path", ""))
        if not filepath and isinstance(result, dict):
            filepath = result.get("path", result.get("file", ""))

        if filepath:
            p = Path(filepath)
            if p.exists():
                size = p.stat().st_size
                if size == 0:
                    vr.passed = False
                    vr.issues.append(f"文件已创建但内容为空: {filepath}")
                    vr.suggestions.append("重新写入完整内容")
                    vr.severity = "warning"
                # 不检查太大或太小——交给语义验证
            else:
                vr.passed = False
                vr.issues.append(f"文件未成功创建: {filepath}")
                vr.suggestions.append(f"检查路径权限，确保目录存在")
                vr.severity = "error"

        # 检查 result 中是否有错误描述
        result_str = self._to_str(result)
        if "failed" in result_str.lower() or "error" in result_str.lower():
            vr.passed = False
            vr.issues.append("写入结果中包含错误信息")
            vr.severity = "warning"

    def _verify_file_read(self, result, vr):
        """验证文件读取——内容不应为空。"""
        if self._is_empty(result):
            vr.passed = False
            vr.issues.append("文件读取返回空内容——文件可能不存在或为空")
            vr.suggestions.append("检查文件路径是否正确")
            vr.severity = "error"

    def _verify_web(self, result, vr):
        """验证 HTTP 请求结果。"""
        result_str = self._to_str(result)
        if not result_str:
            vr.passed = False
            vr.issues.append("HTTP 请求返回空响应")
            vr.suggestions.append("检查 URL 是否正确，网络是否可达")
            vr.severity = "error"
        elif "timeout" in result_str.lower():
            vr.passed = False
            vr.issues.append("HTTP 请求超时")
            vr.suggestions.append("增加超时时间或检查服务端")
            vr.severity = "warning"

    # ═══ 错误分析 ═══

    def _analyze_shell_error(self, output: str) -> str:
        """分析 shell 错误并给出具体修正建议。"""
        output_lower = output.lower()

        # 更具体的模式放前面，避免被泛模式抢先匹配
        patterns = [
            (r"docker.*not found|docker.*not running|docker.*cannot connect", "Docker 未运行——启动 Docker Desktop"),
            (r"npm ERR!|ERESOLVE", "npm 依赖冲突——尝试 npm install --legacy-peer-deps"),
            (r"pip.*error|pip.*ReadTimeout|Could not find.*package", "pip 安装失败——检查包名、Python 版本、尝试换镜像源"),
            (r"ModuleNotFoundError|no module named", "Python 模块未安装——先运行 pip install"),
            (r"port.*already in use|address already in use", "端口被占用——换端口或先 kill 占用进程"),
            (r"not a git repository", "不是 Git 仓库——先运行 git init"),
            (r"failed to push|push.*rejected", "推送被拒——先 git pull 同步远程变更"),
            (r"merge conflict", "合并冲突——手动解决冲突后 commit"),
            (r"command\s+not\s+found", "命令不存在——检查是否已安装，或使用正确的命令名"),
            (r"permission\s+denied", "权限不足——尝试 sudo 或检查文件权限"),
            (r"no\s+such\s+file", "文件不存在——检查路径是否正确"),
            (r"SyntaxError|invalid syntax", "Python 语法错误——检查代码语法"),
            (r"connection\s+refused|connect.*refused", "连接被拒绝——检查服务是否启动、端口是否正确"),
            (r"timeout|timed out|ReadTimeout", "操作超时——增加超时时间或检查网络"),
            (r"out of memory|OOM", "内存不足——释放内存或减少数据量"),
            (r"no space left", "磁盘空间不足——清理磁盘"),
            (r"access denied|authentication failed|unauthorized", "认证失败——检查 token/密码是否正确"),
            (r"cannot\s+find|not\s+found", "找不到目标——检查路径或包名"),
        ]

        for pattern, suggestion in patterns:
            if re.search(pattern, output_lower):
                return suggestion
        return ""

    # ═══ 工具方法 ═══

    @staticmethod
    def _to_str(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("stdout", result.get("output", str(result)))
        return str(result)

    @staticmethod
    def _is_empty(result: Any) -> bool:
        if result is None:
            return True
        if isinstance(result, str):
            return len(result.strip()) == 0
        if isinstance(result, dict):
            stdout = result.get("stdout", "")
            return not stdout or len(stdout.strip()) == 0
        return False
