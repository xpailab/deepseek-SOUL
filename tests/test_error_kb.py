"""错误知识库测试。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from soul.memory.error_kb import ErrorEntry, ErrorKnowledgeBase


@pytest.fixture
def temp_kb_path():
    """临时知识库路径——测试隔离，不污染 ~/.soul/error_knowledge.json。"""
    tmp = tempfile.mkdtemp(prefix="soul_kb_test_")
    yield Path(tmp) / "test_kb.json"
    shutil.rmtree(tmp, ignore_errors=True)


def _new_kb(path: str) -> ErrorKnowledgeBase:
    kb = ErrorKnowledgeBase(path)
    kb.load()
    return kb


class TestErrorEntry:
    def test_create(self):
        e = ErrorEntry(
            signature="abc123",
            pattern="docker.*error",
            tool="bash",
            root_cause="Docker daemon not running",
            fix="Start Docker Desktop",
        )
        assert e.signature == "abc123"
        assert e.confidence == 0.5

    def test_confidence_increases(self):
        e = ErrorEntry(signature="x", pattern=".*", tool="bash", root_cause="", fix="")
        e.usage_count = 10
        e.success_count = 8
        assert e.confidence == 0.8

    def test_stale(self):
        import time
        e = ErrorEntry(signature="x", pattern=".*", tool="bash", root_cause="", fix="")
        e.created_at = time.time() - 40 * 86400
        assert e.is_stale


class TestErrorKnowledgeBase:
    def test_load_save_roundtrip(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kb.learn(
            error_text="docker: Cannot connect to the Docker daemon",
            tool="bash",
            root_cause="Docker daemon not running",
            fix="Start Docker Desktop via systemctl or GUI",
        )
        kb.save()
        assert temp_kb_path.exists()

        kb2 = _new_kb(str(temp_kb_path))
        assert len(kb2.entries) == 1
        entry = list(kb2.entries.values())[0]
        assert "Docker" in entry.root_cause or "Docker" in entry.fix

    def test_lookup_by_signature(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kb.learn(error_text="command not found: nonexist", tool="bash", fix="Install the package")
        entry = kb.lookup("command not found: nonexist")
        assert entry is not None

    def test_lookup_by_pattern(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kb.learn(
            error_text="docker: error during connect",
            tool="bash",
            pattern=r"docker.*error.*connect",
            fix="Start Docker",
        )
        entry = kb.lookup("docker: error during connect to daemon")
        assert entry is not None

    def test_lookup_not_found(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        assert kb.lookup("completely unknown error xyz123") is None

    def test_confidence_filter(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kb.learn(error_text="test error", tool="bash", fix="some fix")
        kb.record_result("test error", success=True)
        kb.record_result("test error", success=True)
        assert kb.lookup_by_confidence("test error", min_confidence=0.8) is not None
        assert kb.lookup_by_confidence("nonexistent", min_confidence=0.8) is None

    def test_record_result(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kb.learn(error_text="error A", tool="bash", fix="fix A")
        kb.record_result("error A", success=True)
        kb.record_result("error A", success=True)
        entry = kb.lookup("error A")
        assert entry.success_count == 2

    def test_signature_normalization(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        sig1 = kb._make_signature("Connection refused to 192.168.1.100:8080")
        sig2 = kb._make_signature("Connection refused to 10.0.0.1:3000")
        assert sig1 == sig2

    def test_keyword_extraction(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kw = kb._extract_keywords("docker container failed to start error")
        assert "docker" in kw
        assert "container" in kw
        assert "failed" in kw
        assert "the" not in kw

    def test_get_stats(self, temp_kb_path):
        kb = _new_kb(str(temp_kb_path))
        kb.learn(error_text="e1", tool="bash", fix="f1")
        kb.learn(error_text="e2", tool="bash", fix="f2")
        kb.lookup("e1")
        kb.lookup("nonexistent")
        stats = kb.get_stats()
        assert stats["total_entries"] == 2
        assert stats["total_lookups"] == 2
        assert stats["total_hits"] == 1

    def test_prune_stale(self, temp_kb_path):
        import time
        kb = _new_kb(str(temp_kb_path))
        kb.MAX_ENTRIES = 3
        e = kb.learn(error_text="old error", tool="bash", fix="old fix")
        e.created_at = time.time() - 40 * 86400
        e.last_used = 0
        for i in range(5):
            kb.learn(error_text=f"new error {i}", tool="bash", fix=f"fix {i}")
        assert len(kb.entries) <= kb.MAX_ENTRIES
