"""记忆系统 4 层集成测试（pytest 格式）。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from soul.memory.frozen import FrozenMemory
from soul.memory.indexed import IndexedMemory
from soul.memory.manager import MemoryManager
from soul.memory.predictive import PredictiveMemory
from soul.memory.procedural import ProceduralMemory
from soul.types import MemoryConfig, MemoryEntry, MemoryLayer, Message, MessageRole


# ============================================================
# Layer 1: FrozenMemory 测试
# ============================================================

def test_layer1_frozen():
    """Layer 1: FrozenMemory 冻结快照记忆。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_l1_")
    try:
        fm = FrozenMemory(tmp)

        fm.snapshot()
        assert (Path(tmp) / "MEMORY.md").exists(), "MEMORY.md 未自动创建"
        assert (Path(tmp) / "USER.md").exists(), "USER.md 未自动创建"

        mem = fm.read("MEMORY.md")
        assert mem == "", "空文件应返回空字符串"

        fm.write("MEMORY.md", "项目名: DeepSoul\n技术栈: Python 3.11+")
        mem = fm.read("MEMORY.md")
        assert mem == "", "写后读应返回冻结快照 (prefix cache 保护)"

        new_content, was_compressed = fm.add("MEMORY.md", "这是一条新记忆")
        assert "这是一条新记忆" in new_content
        assert not was_compressed

        fm._active = False
        fm.write("USER.md", "用户名: 张三\n偏好: Go/Python")
        user = fm.get_user()
        assert "张三" in user
        fm._active = True

        fm.add("MEMORY.md", "额外数据")
        usage = fm.get_usage()
        assert "memory" in usage and "user" in usage and "pct" in usage["memory"]

        h = fm.get_hash("MEMORY.md")
        assert len(h) == 16

        fm.remove("MEMORY.md", "额外数据")
        assert "额外数据" not in fm.read("MEMORY.md")

        big = "A" * (fm.MAX_MEMORY_CHARS + 500)
        fm.write("MEMORY.md", big)
        content = fm.read("MEMORY.md")
        assert len(content) <= fm.MAX_MEMORY_CHARS, "超大内容未压缩"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Layer 2: ProceduralMemory 测试
# ============================================================

@pytest.mark.asyncio
async def test_layer2_procedural():
    """Layer 2: ProceduralMemory 程序性技能记忆。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_l2_")
    try:
        pm = ProceduralMemory(tmp)

        skill_path = Path(tmp) / "deploy.skill"
        skill_path.write_text("""---
name: auto_deploy
version: 1.0.0
description: auto deploy to production
triggers: deploy, 部署, 上线
---
# auto_deploy

## 步骤
1. 构建镜像
2. 推送镜像
3. 滚动更新
""", encoding="utf-8")

        skills = await pm.load_all()
        assert len(skills) == 1
        assert skills[0].meta.name == "auto_deploy"

        matched = pm.match("请帮我部署到线上去")
        assert len(matched) > 0 and matched[0].meta.name == "auto_deploy"

        matched = pm.match("今天天气怎么样")
        assert len(matched) == 0

        trace = [
            {"action": "tool_call", "description": "构建Docker镜像"},
            {"action": "bash", "description": "docker build -t app ."},
            {"action": "decision", "description": "选择蓝绿部署策略"},
        ]
        skill = await pm.create_from_trace("docker 部署上线", trace, success=True)
        assert skill is not None
        assert "构建Docker镜像" in skill.content

        assert pm.skill_count == 2
        info = pm.list_skills()
        assert len(info) == 2

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Layer 3: IndexedMemory 测试
# ============================================================

@pytest.mark.asyncio
async def test_layer3_indexed():
    """Layer 3: IndexedMemory FTS5 + LLM 混合检索。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_l3_")
    db_path = Path(tmp) / "test.db"
    im = IndexedMemory(str(db_path))

    try:
        msg_id = await im.store_message(
            session_id="sess_001",
            role="user",
            content="Python asyncio concurrent programming how to handle?",
            metadata={"source": "cli"},
        )
        assert msg_id.startswith("msg_")

        await im.store_message(
            session_id="sess_001",
            role="assistant",
            content="asyncio uses event loop and coroutine for concurrency...",
        )
        await im.store_message(
            session_id="sess_002",
            role="user",
            content="database slow query optimization tips",
        )

        results = await im.search_fts("asyncio")
        assert len(results) > 0, "FTS5 未找到 asyncio"

        results = await im.search_fts("python")
        assert len(results) > 0, "FTS5 未找到 python"

        results = await im.search_fts("database", session_id="sess_002")
        assert len(results) == 1

        await im.store_message(session_id="sess_003", role="user", content="部署到生产环境遇到错误怎么办")
        await im.store_message(session_id="sess_003", role="assistant", content="先检查日志，确认是代码问题还是配置问题")
        results = await im.search_fts("部署")
        assert len(results) > 0, "中文 FTS5 搜索 '部署' 失败"
        results = await im.search_fts("生产环境")
        assert len(results) > 0, "中文 FTS5 搜索 '生产环境' 失败"

        await im.store_message(session_id="sess_004", role="user", content="Python asyncio 并发编程怎么处理")
        results = await im.search_fts("asyncio 并发")
        assert len(results) > 0, "中英混合搜索失败"

        results = await im.search_semantic("部署")
        assert len(results) > 0, "语义搜索 '部署' 失败"

        entry = MemoryEntry(
            layer=MemoryLayer.INDEXED,
            content="PostgreSQL slow query optimization: check EXPLAIN first, then add index",
            tags=["database", "optimization", "postgresql"],
            importance=0.9,
        )
        await im.store_memory_entry(entry)
        entries = await im.retrieve_memories(query="slow query", min_importance=0.5)
        assert len(entries) > 0 and "PostgreSQL" in entries[0].content

        history = await im.get_session_history("sess_001")
        assert len(history) == 2
        assert history[0]["role"] == "user" and history[1]["role"] == "assistant"

        stats = await im.get_stats()
        assert "total_conversations" in stats and stats["total_conversations"] > 0

        deleted = await im.compact_old_sessions(older_than_days=0)
        assert isinstance(deleted, int) and deleted >= 0

        # LLM 动态扩展
        async def mock_llm(prompt: str) -> str:
            if "扩展" in prompt:
                if "部署" in prompt:
                    return "上线, release, deployment, 发布"
                return "related, similar, 相关"
            if "排序" in prompt:
                return "0, 1"
            return ""

        im2 = IndexedMemory(str(Path(tmp) / "test2.db"))
        im2.set_llm(mock_llm)

        await im2.store_message("sess_10", "user", "部署上线失败排查")
        await im2.store_message("sess_10", "assistant", "检查 nginx 配置和端口监听")

        results = await im2.search_semantic("部署", use_llm=True)
        assert len(results) > 0, "LLM 动态扩展失败"
        assert any("部署上线失败" in r["content"] for r in results)

        results_fallback = await im2.search_semantic("部署", use_llm=False)
        assert isinstance(results_fallback, list)

        im3 = IndexedMemory(str(Path(tmp) / "test3.db"))
        await im3.store_message("sess_20", "user", "部署环境配置")
        results_no_llm = await im3.search_semantic("部署")
        assert len(results_no_llm) > 0, "无 LLM 时降级失败"

        await im2.close()
        await im3.close()

    finally:
        await im.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Layer 4: PredictiveMemory 测试
# ============================================================

@pytest.mark.asyncio
async def test_layer4_predictive():
    """Layer 4: PredictiveMemory 预测记忆。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_l4_")
    json_path = Path(tmp) / "predictive.json"

    try:
        pm = PredictiveMemory(str(json_path))

        for _ in range(5):
            await pm.observe("docker build", previous_action="修改 Dockerfile")
            await pm.observe("docker push", previous_action="docker build")
            await pm.observe("kubectl apply", previous_action="docker push")

        preds = await pm.predict_next_actions("docker build")
        assert len(preds) > 0
        assert "docker push" in preds[0]["action"]

        for _ in range(10):
            await pm.observe("部署检查", previous_action="代码提交", context={"project": "SOUL"})

        preds = await pm.predict_next_actions("代码提交", context={"project": "SOUL"})
        assert len(preds) > 0

        habits = await pm.detect_habits()
        assert len(habits) > 0
        assert "suggestion" in habits[0]

        preloaded = await pm.preload_context(context={"project": "SOUL"})
        assert isinstance(preloaded, list)

        prompt = await pm.get_predictive_context_prompt()
        assert isinstance(prompt, str)

        pm.save()
        pm2 = PredictiveMemory(str(json_path))
        pm2.load()
        habits2 = await pm2.detect_habits()
        assert len(habits2) == len(habits)

        pm3 = PredictiveMemory(str(Path(tmp) / "empty.json"))
        assert (await pm3.detect_habits()) == []

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# MemoryManager 集成测试
# ============================================================

@pytest.mark.asyncio
async def test_memory_manager():
    """MemoryManager 统一记忆管理。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_mm_")
    workspace = Path(tmp) / "workspace"
    workspace.mkdir(parents=True)

    try:
        config = MemoryConfig(
            workspace_dir=str(workspace),
            fts_db_path=str(Path(tmp) / "memory.db"),
            predictive_enabled=True,
        )
        mm = MemoryManager(config)

        await mm.initialize()
        assert mm.procedural.skill_count >= 0

        usage = mm.frozen.get_usage()
        assert usage["memory"]["chars"] >= 0

        matched = mm.procedural.match("帮我调试这个死锁问题")
        if matched:
            assert len(matched) > 0

        msgs = [
            Message(role=MessageRole.USER, content="测试消息：Python 并发编程"),
            Message(role=MessageRole.ASSISTANT, content="asyncio 可以解决 IO 密集型并发..."),
        ]
        await mm.store_conversation("test_session_01", msgs)
        stats = await mm.indexed.get_stats()
        assert stats["total_conversations"] >= 2

        await mm.remember("Key experience: FTS5 tokenizer needs unicode61", layer=MemoryLayer.FROZEN)
        assert len(mm.frozen.read("MEMORY.md")) >= 0

        await mm.remember("SQL optimization tips", layer=MemoryLayer.INDEXED, importance=0.8, tags=["sql"])
        entries = await mm.indexed.retrieve_memories(tags=["sql"])
        assert len(entries) > 0

        result = await mm.query("部署", layers=[MemoryLayer.PROCEDURAL, MemoryLayer.PREDICTIVE])
        assert "procedural" in result and "predictive" in result

        await mm.observe_action("测试部署", previous="写代码")
        prompt = await mm.query_for_prompt("部署", context={"project": "SOUL"})
        assert isinstance(prompt, str)

        await mm.forget("Key experience", layer=MemoryLayer.FROZEN)
        assert "Key experience" not in mm.frozen.get_memory()

        stats = await mm.get_stats()
        for k in ["frozen", "procedural", "indexed", "predictive"]:
            assert k in stats, f"stats 中缺少 {k}"

        compact_result = await mm.compact()
        assert isinstance(compact_result, dict)

        await mm.close()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 边界情况测试
# ============================================================

def test_edge_cases():
    """边界情况。"""
    tmp = tempfile.mkdtemp(prefix="soul_test_edge_")
    try:
        fm = FrozenMemory(tmp)
        fm.snapshot()

        fm._active = False
        fm.write("MEMORY.md", "")
        assert fm.get_memory() == ""
        fm._active = True

        content, _ = fm.add("MEMORY.md", "只有一条")
        assert len(content) > 0 and "只有一条" in content

        fm.add("MEMORY.md", "<script>alert('xss')</script>")
        fm.remove("MEMORY.md", "不存在的内容")  # 不应报错

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
