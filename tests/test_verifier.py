"""结果验证器测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from soul.engine.verifier import ResultVerifier, VerifyResult


class TestVerifyResult:
    def test_passed_empty_context(self):
        vr = VerifyResult(passed=True, tool_name="bash")
        assert vr.to_prompt_context() == ""

    def test_failed_context(self):
        vr = VerifyResult(
            passed=False,
            tool_name="bash",
            issues=["输出为空"],
            suggestions=["重新执行命令"],
        )
        ctx = vr.to_prompt_context()
        assert "验证失败" in ctx
        assert "输出为空" in ctx
        assert "重新执行命令" in ctx


class TestResultVerifier:
    def test_empty_result(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", None, "")
        assert not vr.passed
        assert any("空" in i for i in vr.issues)

    def test_shell_success(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "hello world", "")
        assert vr.passed

    def test_shell_with_traceback(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "Traceback (most recent call last):\n  File test.py\nError: something", "")
        assert not vr.passed
        assert any("Traceback" in i for i in vr.issues)

    def test_shell_with_command_not_found(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "bash: nonexist: command not found", "")
        assert not vr.passed

    def test_shell_with_permission_denied(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "Permission denied: cannot access /root", "")
        assert not vr.passed

    def test_shell_with_error_in_stderr(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "some output", "Error: connection refused")
        assert not vr.passed

    def test_shell_nonzero_exit(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "output", "", exit_code=1)
        assert not vr.passed
        assert any("退出码" in i or "1" in i for i in vr.issues)

    def test_shell_normal_output(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("bash", "Package installed successfully", "")
        assert vr.passed

    def test_file_write_creates_empty_file(self):
        v = ResultVerifier()
        tmp = tempfile.mkdtemp(prefix="soul_vrf_")
        empty_file = Path(tmp) / "empty.txt"
        empty_file.write_text("")

        vr = v.verify_tool_result(
            "write_file", "file written",
            context={"file_path": str(empty_file)},
        )
        assert not vr.passed
        assert any("空" in i for i in vr.issues)

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_file_write_success(self):
        v = ResultVerifier()
        tmp = tempfile.mkdtemp(prefix="soul_vrf_")
        good_file = Path(tmp) / "good.txt"
        good_file.write_text("content here")

        vr = v.verify_tool_result(
            "write_file", "file written",
            context={"file_path": str(good_file)},
        )
        assert vr.passed

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_file_read_empty(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("read_file", "", "")
        assert not vr.passed

    def test_file_read_with_content(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("read_file", "file contents here", "")
        assert vr.passed

    def test_web_empty_response(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("web", "", "")
        assert not vr.passed

    def test_web_timeout(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("web", "request timeout after 30s", "")
        assert not vr.passed

    def test_web_ok(self):
        v = ResultVerifier()
        vr = v.verify_tool_result("web", '{"status": "ok", "data": []}', "")
        assert vr.passed


class TestErrorAnalysis:
    def test_command_not_found(self):
        v = ResultVerifier()
        s = v._analyze_shell_error("bash: python3: command not found")
        assert "命令不存在" in s or "未安装" in s or "正确的命令" in s

    def test_permission_denied(self):
        v = ResultVerifier()
        s = v._analyze_shell_error("Permission denied: /etc/passwd")
        assert "权限" in s

    def test_port_in_use(self):
        v = ResultVerifier()
        s = v._analyze_shell_error("Error: address already in use")
        assert "端口" in s or "占用" in s

    def test_pip_error(self):
        v = ResultVerifier()
        s = v._analyze_shell_error("pip._vendor.urllib3.exceptions.ReadTimeoutError")
        assert "pip" in s.lower() and ("失败" in s or "超时" in s or "镜像" in s)

    def test_docker_not_running(self):
        v = ResultVerifier()
        s = v._analyze_shell_error("docker: Cannot connect to the Docker daemon")
        assert "Docker" in s

    def test_no_match(self):
        v = ResultVerifier()
        s = v._analyze_shell_error("some random text without known patterns")
        assert s == ""
