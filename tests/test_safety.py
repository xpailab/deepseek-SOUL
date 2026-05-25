"""安全系统测试：auditor, sandbox。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from soul.safety.auditor import Auditor
from soul.safety.sandbox import Sandbox


class TestAuditor:
    def test_record_and_flush(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        auditor.record("test_event", {"key": "value"})
        assert len(auditor._buffer) == 1
        auditor.flush()
        assert len(auditor._buffer) == 0
        # 验证日志文件已创建
        import time
        date_str = time.strftime("%Y-%m-%d")
        log_file = Path(tmp) / f"audit_{date_str}.jsonl"
        assert log_file.exists()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_record_tool_call(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        auditor.record_tool_call("bash", {"command": "echo hello"}, "hello", "session_1")
        assert len(auditor._buffer) == 1
        assert auditor._buffer[0]["type"] == "tool_call"
        auditor.flush()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_record_safety_block(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        auditor.record_safety_block("bash", "rm -rf is dangerous", {"command": "rm -rf /"})
        assert len(auditor._buffer) == 1
        assert auditor._buffer[0]["severity"] == "warning"
        auditor.flush()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_record_file_access(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        auditor.record_file_access("/tmp/test.txt", "write", "session_1")
        assert auditor._buffer[0]["type"] == "file_access"
        auditor.flush()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_auto_flush_on_buffer_full(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        for i in range(auditor._buffer_size):
            auditor.record("event", {"i": i})
        assert len(auditor._buffer) == 0  # 自动刷新
        shutil.rmtree(tmp, ignore_errors=True)

    def test_query(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        auditor.record("tool_call", {"tool": "bash"}, severity="info")
        auditor.record("safety_block", {"tool": "bash"}, severity="warning")
        auditor.flush()

        events = auditor.query(event_type="tool_call")
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"

        events = auditor.query(severity="warning")
        assert len(events) == 1
        assert events[0]["type"] == "safety_block"

        empty = auditor.query(event_type="nonexistent")
        assert len(empty) == 0

        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_security_report(self):
        tmp = tempfile.mkdtemp(prefix="soul_audit_")
        auditor = Auditor(log_dir=tmp)
        auditor.record_tool_call("bash", {}, "ok", "s1")
        auditor.record_safety_block("rm", "blocked", {})
        auditor.flush()

        report = auditor.get_security_report()
        assert "total_events" in report
        assert "safety_blocks" in report
        assert report["safety_blocks"] == 1
        assert "block_rate" in report

        shutil.rmtree(tmp, ignore_errors=True)


class TestSandbox:
    @pytest.mark.asyncio
    async def test_local_execute_echo(self):
        s = Sandbox()
        result = await s.execute(command="echo hello", mode="local", timeout=10)
        assert "hello" in result.get("stdout", "")

    @pytest.mark.asyncio
    async def test_local_execute_with_stderr(self):
        s = Sandbox()
        result = await s.execute(command="echo ok && echo err >&2", mode="local", timeout=10)
        assert "ok" in result.get("stdout", "")

    @pytest.mark.asyncio
    async def test_local_execute_timeout(self):
        s = Sandbox()
        result = await s.execute(command="sleep 10", mode="local", timeout=1)
        # 超时后应返回结果
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_local_execute_invalid_command(self):
        s = Sandbox()
        result = await s.execute(command="nonexistent_command_xyz", mode="local", timeout=10)
        assert isinstance(result, dict)
