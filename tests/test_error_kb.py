"""错误知识库测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from soul.memory.error_kb import ErrorEntry, ErrorKnowledgeBase


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
        assert e.confidence == 0.5  # 未使用过，默认 0.5

    def test_confidence_increases(self):
        e = ErrorEntry(signature="x", pattern=".*", tool="bash", root_cause="", fix="")
        e.usage_count = 10
        e.success_count = 8
        assert e.confidence == 0.8

    def test_stale(self):
        import time
        e = ErrorEntry(signature="x", pattern=".*", tool="bash", root_cause="", fix="")
        e.created_at = time.time() - 40 * 86400  # 40 天前
        assert e.is_stale  # property, 默认 30 天


class TestErrorKnowledgeBase:
    def test_load_save_roundtrip(self):
        tmp = tempfile.mkdtemp(prefix="soul_kb_")
        kb_path = Path(tmp) / "test_kb.json"

        kb = ErrorKnowledgeBase(str(kb_path))
        kb.load()
        kb.learn(
            error_text="docker: Cannot connect to the Docker daemon",
            tool="bash",
            root_cause="Docker daemon not running",
            fix="Start Docker Desktop via systemctl or GUI",
        )
        kb.save()
        assert kb_path.exists()

        # 重新加载
        kb2 = ErrorKnowledgeBase(str(kb_path))
        kb2.load()
        assert len(kb2.entries) == 1
        entry = list(kb2.entries.values())[0]
        assert "Docker" in entry.root_cause or "Docker" in entry.fix

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_lookup_by_signature(self):
        kb = ErrorKnowledgeBase()
        kb.load()
        kb.learn(
            error_text="command not found: nonexist",
            tool="bash",
            fix="Install the package",
        )
        entry = kb.lookup("command not found: nonexist")
        assert entry is not None

    def test_lookup_by_pattern(self):
        kb = ErrorKnowledgeBase()
        kb.load()
        kb.learn(
            error_text="docker: error during connect",
            tool="bash",
            pattern=r"docker.*error.*connect",
            fix="Start Docker",
        )
        entry = kb.lookup("docker: error during connect to daemon")
        assert entry is not None

    def test_lookup_not_found(self):
        kb = ErrorKnowledgeBase()
        kb.load()
        assert kb.lookup("completely unknown error xyz123") is None

    def test_confidence_filter(self):
        kb = ErrorKnowledgeBase()
        kb.load()
        kb.learn(error_text="test error", tool="bash", fix="some fix")
        # 学习后记录成功结果——提升置信度
        kb.record_result("test error", success=True)
        kb.record_result("test error", success=True)
        # 2/2 = 1.0 置信度
        assert kb.lookup_by_confidence("test error", min_confidence=0.8) is not None
        assert kb.lookup_by_confidence("nonexistent", min_confidence=0.8) is None

    def test_record_result(self):
        kb = ErrorKnowledgeBase()
        kb.load()
        kb.learn(error_text="error A", tool="bash", fix="fix A")
        kb.record_result("error A", success=True)
        kb.record_result("error A", success=True)
        entry = kb.lookup("error A")
        assert entry.success_count == 2

    def test_signature_normalization(self):
        kb = ErrorKnowledgeBase()
        # 相同语义的错误应有相同签名
        sig1 = kb._make_signature("Connection refused to 192.168.1.100:8080")
        sig2 = kb._make_signature("Connection refused to 10.0.0.1:3000")
        assert sig1 == sig2  # IP 和端口被标准化

    def test_keyword_extraction(self):
        kb = ErrorKnowledgeBase()
        kw = kb._extract_keywords("docker container failed to start error")
        assert "docker" in kw
        assert "container" in kw
        assert "failed" in kw
        assert "the" not in kw  # 停用词

    def test_get_stats(self):
        kb = ErrorKnowledgeBase()
        kb.load()
        kb.learn(error_text="e1", tool="bash", fix="f1")
        kb.learn(error_text="e2", tool="bash", fix="f2")
        kb.lookup("e1")
        kb.lookup("nonexistent")
        stats = kb.get_stats()
        assert stats["total_entries"] == 2
        assert stats["total_lookups"] == 2
        assert stats["total_hits"] == 1

    def test_prune_stale(self):
        import time

        kb = ErrorKnowledgeBase()
        kb.load()
        kb.MAX_ENTRIES = 3
        # 添加过期条目
        e = kb.learn(error_text="old error", tool="bash", fix="old fix")
        e.created_at = time.time() - 40 * 86400
        e.last_used = 0  # 从未使用
        # 添加新鲜条目
        for i in range(5):
            kb.learn(error_text=f"new error {i}", tool="bash", fix=f"fix {i}")
        # 应裁剪到 MAX_ENTRIES
        assert len(kb.entries) <= kb.MAX_ENTRIES
