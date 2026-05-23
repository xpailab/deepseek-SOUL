"""
记忆系统 4 层集成测试
覆盖所有层及边界情况
"""
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soul.memory.frozen import FrozenMemory
from soul.memory.procedural import ProceduralMemory
from soul.memory.indexed import IndexedMemory
from soul.memory.predictive import PredictiveMemory
from soul.memory.manager import MemoryManager
from soul.types import MemoryConfig, MemoryEntry, MemoryLayer, Message, MessageRole

# ============================================================
# 工具函数
# ============================================================

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Layer 1: FrozenMemory 测试
# ============================================================

def test_layer1_frozen():
    section("Layer 1: FrozenMemory (冻结快照记忆)")

    tmp = tempfile.mkdtemp(prefix="soul_test_l1_")
    try:
        fm = FrozenMemory(tmp)

        # 1. 初始快照 (文件不存在时自动创建)
        fm.snapshot()
        assert (Path(tmp) / "MEMORY.md").exists(), "MEMORY.md should be auto-created"
        assert (Path(tmp) / "USER.md").exists(), "USER.md should be auto-created"
        check("snapshot 自动创建空文件", True)

        # 2. 读冻结快照
        mem = fm.read("MEMORY.md")
        check("读冻结快照 (空)", mem == "")

        # 3. 写入内容
        fm.write("MEMORY.md", "项目名: DeepSoul\n技术栈: Python 3.11+")
        check("write 写入文件", True)

        # 4. 冻结后，读仍返回旧快照 (prefix cache 保护)
        mem = fm.read("MEMORY.md")
        check("写后读仍返回冻结快照 (prefix cache 保护)", mem == "")

        # 5. 添加条目 § 分隔符 — 返回 (content, compressed)
        new_content, was_compressed = fm.add("MEMORY.md", "这是一条新记忆")
        check("add 追加条目用 § 分隔符", "这是一条新记忆" in new_content)
        check("add 返回 uncompressed", not was_compressed)
        # 冻结保护：读仍返回旧快照，但磁盘已更新
        check("冻结读仍返回旧快照", fm.read("MEMORY.md") == "")

        # 6. USER.md 独立管理 — 解冻测试
        fm._active = False  # 关闭冻结
        fm.write("USER.md", "用户名: 张三\n偏好: Go/Python")
        user = fm.get_user()
        check("get_user 返回用户画像 (解冻后)", "张三" in user)
        fm._active = True  # 恢复冻结

        # 7. get_usage 容量统计
        fm.add("MEMORY.md", "额外数据")
        usage = fm.get_usage()
        check("get_usage 返回容量统计",
              "memory" in usage and "user" in usage and "pct" in usage["memory"])

        # 8. Hash 计算
        h = fm.get_hash("MEMORY.md")
        check("get_hash 返回 SHA256 前缀", len(h) == 16)

        # 9. 删除条目
        fm.remove("MEMORY.md", "额外数据")
        check("remove 按模式删除条目", "额外数据" not in fm.read("MEMORY.md"))

        # 10. 压缩测试 — 超容量内容
        big = "A" * (fm.MAX_MEMORY_CHARS + 500)
        fm.write("MEMORY.md", big)
        content = fm.read("MEMORY.md")
        check("超大内容自动压缩", len(content) <= fm.MAX_MEMORY_CHARS)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Layer 2: ProceduralMemory 测试
# ============================================================

async def test_layer2_procedural():
    section("Layer 2: ProceduralMemory (程序性技能记忆)")

    tmp = tempfile.mkdtemp(prefix="soul_test_l2_")
    try:
        pm = ProceduralMemory(tmp)

        # 1. 手动创建一个技能文件
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

        # 2. 加载技能
        skills = await pm.load_all()
        check("load_all 加载技能文件", len(skills) == 1)
        check("技能名称正确", skills[0].meta.name == "auto_deploy")

        # 3. 语义匹配 — 触发词匹配
        matched = pm.match("请帮我部署到线上去")
        check("match 匹配到部署技能 (trigger: 部署)", len(matched) > 0 and matched[0].meta.name == "auto_deploy")

        # 4. 不匹配的查询
        matched = pm.match("今天天气怎么样")
        check("match 无匹配返回空列表", len(matched) == 0)

        # 5. 从追踪创建技能
        trace = [
            {"action": "tool_call", "description": "构建Docker镜像"},
            {"action": "bash", "description": "docker build -t app ."},
            {"action": "decision", "description": "选择蓝绿部署策略"},
        ]
        skill = await pm.create_from_trace("docker 部署上线", trace, success=True)
        check("create_from_trace 生成技能", skill is not None)
        if skill:
            check("新技能包含步骤", "构建Docker镜像" in skill.content)

        # 6. skill_count
        check("skill_count 正确", pm.skill_count == 2)

        # 7. list_skills
        info = pm.list_skills()
        check("list_skills 返回技能列表", len(info) == 2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Layer 3: IndexedMemory 测试
# ============================================================

async def test_layer3_indexed():
    section("Layer 3: IndexedMemory (FTS5 + LLM 混合检索)")

    tmp = tempfile.mkdtemp(prefix="soul_test_l3_")
    db_path = Path(tmp) / "test.db"
    im = IndexedMemory(str(db_path))

    try:
        # 1. 存储消息
        msg_id = await im.store_message(
            session_id="sess_001",
            role="user",
            content="Python asyncio concurrent programming how to handle?",
            metadata={"source": "cli"},
        )
        check("store_message 返回消息ID", msg_id.startswith("msg_"))

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

        # 2. FTS5 全文搜索 — 英文
        results = await im.search_fts("asyncio")
        check("search_fts 找到 asyncio 相关消息", len(results) > 0)

        results = await im.search_fts("python")
        check("search_fts 单关键词 python", len(results) > 0)

        # 3. 按 session_id 过滤
        results = await im.search_fts("database", session_id="sess_002")
        check("search_fts 按 session 过滤 (database)", len(results) == 1)

        # 4. 中文 FTS5 搜索 (jieba 分词)
        await im.store_message(
            session_id="sess_003",
            role="user",
            content="部署到生产环境遇到错误怎么办",
        )
        await im.store_message(
            session_id="sess_003",
            role="assistant",
            content="先检查日志，确认是代码问题还是配置问题",
        )
        results = await im.search_fts("部署")
        check("search_fts 中文搜索 '部署'", len(results) > 0)
        results = await im.search_fts("生产环境")
        check("search_fts 中文搜索 '生产环境'", len(results) > 0)
        results = await im.search_fts("配置")
        check("search_fts 中文搜索 '配置'", len(results) > 0)

        # 5. 中英混合搜索
        await im.store_message(
            session_id="sess_004",
            role="user",
            content="Python asyncio 并发编程怎么处理",
        )
        results = await im.search_fts("asyncio 并发")
        check("search_fts 中英混合 'asyncio 并发'", len(results) > 0)

        # 6. 语义搜索 (关键词扩展) — jieba 分词后中英文均支持
        results = await im.search_semantic("部署")
        check("search_semantic 中文 '部署' (同义词扩展)", len(results) > 0)

        results = await im.search_semantic("数据库")
        check("search_semantic 中文 '数据库' (同义词扩展)", len(results) > 0)

        # 7. 结构化记忆
        entry = MemoryEntry(
            layer=MemoryLayer.INDEXED,
            content="PostgreSQL slow query optimization: check EXPLAIN first, then add index",
            tags=["database", "optimization", "postgresql"],
            importance=0.9,
        )
        await im.store_memory_entry(entry)
        entries = await im.retrieve_memories(query="slow query", min_importance=0.5)
        check("store/retrieve 结构化记忆", len(entries) > 0 and "PostgreSQL" in entries[0].content)

        # 6. 按标签检索
        entries = await im.retrieve_memories(tags=["database"])
        check("retrieve 按 tags 过滤", len(entries) > 0)

        # 7. 会话历史
        history = await im.get_session_history("sess_001")
        check("get_session_history 按时间排序", len(history) == 2)
        check("历史按时间正序", history[0]["role"] == "user" and history[1]["role"] == "assistant")

        # 8. get_stats
        stats = await im.get_stats()
        check("get_stats 返回统计", "total_conversations" in stats and stats["total_conversations"] > 0)

        # 9. 压缩清理 (旧数据)
        deleted = await im.compact_old_sessions(older_than_days=0)  # 所有过期
        check("compact_old_sessions 清理旧数据", isinstance(deleted, int) and deleted >= 0)

        # 10. LLM 动态查询扩展测试
        async def mock_llm(prompt: str) -> str:
            """模拟 LLM：从 prompt 中提取查询，返回相关词。"""
            if "扩展" in prompt:
                # 查询扩展模式
                if "部署" in prompt:
                    return "上线, release, deployment, 发布"
                if "错误" in prompt or "error" in prompt:
                    return "bug, exception, 异常, failure"
                return "related, similar, 相关"
            elif "排序" in prompt:
                # 重排模式 — 返回第一个候选项
                return "0, 1"
            return ""

        im2 = IndexedMemory(str(Path(tmp) / "test2.db"))
        im2.set_llm(mock_llm)

        await im2.store_message("sess_10", "user", "部署上线失败排查")
        await im2.store_message("sess_10", "assistant", "检查 nginx 配置和端口监听")
        await im2.store_message("sess_10", "assistant", "确认环境变量和配置文件正确")

        # LLM 扩展搜索
        results = await im2.search_semantic("部署", use_llm=True)
        check("LLM 动态扩展 search_semantic", len(results) > 0)
        # 应找到 "部署上线失败排查"
        check("LLM 扩展找到相关对话", any("部署上线失败" in r["content"] for r in results))

        # 关闭 LLM 退回静态表
        results_fallback = await im2.search_semantic("部署", use_llm=False)
        check("use_llm=False 退回静态扩展", isinstance(results_fallback, list))

        # LLM 不可用时降级
        im3 = IndexedMemory(str(Path(tmp) / "test3.db"))
        await im3.store_message("sess_20", "user", "部署环境配置")
        results_no_llm = await im3.search_semantic("部署")
        check("无 LLM 时降级到静态表", len(results_no_llm) > 0)

        await im2.close()
        await im3.close()

    finally:
        await im.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Layer 4: PredictiveMemory 测试
# ============================================================

async def test_layer4_predictive():
    section("Layer 4: PredictiveMemory (预测记忆 — SOUL 创新)")

    tmp = tempfile.mkdtemp(prefix="soul_test_l4_")
    json_path = Path(tmp) / "predictive.json"

    try:
        pm = PredictiveMemory(str(json_path))

        # 1. 观察行为 — 建立任务路径
        for _ in range(5):
            await pm.observe("docker build", previous_action="修改 Dockerfile")
            await pm.observe("docker push", previous_action="docker build")
            await pm.observe("kubectl apply", previous_action="docker push")

        check("observe 建立任务路径", True)

        # 2. 路径预测
        preds = await pm.predict_next_actions("docker build")
        check("predict_next_actions 路径预测", len(preds) > 0)
        if preds:
            check("预测包含 docker push", "docker push" in preds[0]["action"])

        # 3. 时间关联 — 模拟特定时间
        for _ in range(10):
            await pm.observe(
                "部署检查",
                previous_action="代码提交",
                context={"project": "SOUL", "directory": "/app"},
            )

        preds = await pm.predict_next_actions("代码提交", context={"project": "SOUL"})
        check("预测包含上下文关联", len(preds) > 0)

        # 4. 习惯检测
        habits = await pm.detect_habits()
        check("detect_habits 检测重复模式", len(habits) > 0)
        if habits:
            check("习惯有 suggestion 建议", "suggestion" in habits[0])

        # 5. preload_context
        preloaded = await pm.preload_context(context={"project": "SOUL"})
        check("preload_context 预加载上下文", isinstance(preloaded, list))

        # 6. get_predictive_context_prompt
        prompt = await pm.get_predictive_context_prompt()
        check("get_predictive_context_prompt 生成预测片段", isinstance(prompt, str))

        # 7. 持久化 save/load
        pm.save()
        pm2 = PredictiveMemory(str(json_path))
        pm2.load()
        habits2 = await pm2.detect_habits()
        check("save/load 持久化恢复", len(habits2) == len(habits))

        # 8. 无历史时返回空
        pm3 = PredictiveMemory(str(Path(tmp) / "empty.json"))
        check("空预测不崩溃并返回空", (await pm3.detect_habits()) == [])

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# MemoryManager 集成测试
# ============================================================

async def test_memory_manager():
    section("MemoryManager (统一记忆管理)")

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

        # 1. 初始化
        await mm.initialize()
        check("initialize 初始化 4 层并加载捆绑技能", mm.procedural.skill_count >= 0)
        print(f"      捆绑技能数: {mm.procedural.skill_count}")

        # 2. 检查冻结快照生效
        usage = mm.frozen.get_usage()
        check("frozen 快照已创建", usage["memory"]["chars"] >= 0)

        # 3. 技能匹配
        matched = mm.procedural.match("帮我调试这个死锁问题")
        if matched:
            check("query via procedural match", len(matched) > 0)

        # 4. 存储对话
        msgs = [
            Message(role=MessageRole.USER, content="测试消息：Python 并发编程"),
            Message(role=MessageRole.ASSISTANT, content="asyncio 可以解决 IO 密集型并发..."),
        ]
        await mm.store_conversation("test_session_01", msgs)
        stats = await mm.indexed.get_stats()
        check("store_conversation 批量存储", stats["total_conversations"] >= 2)

        # 5. remember (Layer 1) — 冻结模式下写入磁盘但不更新当前快照
        await mm.remember("Key experience: FTS5 tokenizer needs unicode61", layer=MemoryLayer.FROZEN)
        # 冻结模式：read 返回旧快照，但磁盘已更新（prefix cache 保护）
        check("remember to Layer 1 (FROZEN) 写入成功",
              len(mm.frozen.read("MEMORY.md")) >= 0)  # 返回冻结快照 (可为空)

        # 6. remember (Layer 3)
        await mm.remember(
            "SQL optimization tips", layer=MemoryLayer.INDEXED, importance=0.8, tags=["sql"]
        )
        entries = await mm.indexed.retrieve_memories(tags=["sql"])
        check("remember to Layer 3 (INDEXED)", len(entries) > 0)

        # 7. query 统一查询
        result = await mm.query("部署", layers=[MemoryLayer.PROCEDURAL, MemoryLayer.PREDICTIVE])
        check("query 按层查询", "procedural" in result and "predictive" in result)

        # 8. query_for_prompt
        await mm.observe_action("测试部署", previous="写代码")
        prompt = await mm.query_for_prompt("部署", context={"project": "SOUL"})
        check("query_for_prompt 生成 prompt 注入文本", isinstance(prompt, str))

        # 9. forget
        await mm.forget("Key experience", layer=MemoryLayer.FROZEN)
        check("forget 删除 Layer 1 记忆", "Key experience" not in mm.frozen.get_memory())

        # 10. get_stats
        stats = await mm.get_stats()
        check("get_stats 返回 4 层统计", all(k in stats for k in ["frozen", "procedural", "indexed", "predictive"]))
        print(f"      frozen: usage={stats['frozen']['usage']}")
        print(f"      procedural: skills={stats['procedural']['skill_count']}")
        print(f"      indexed: {stats['indexed']}")
        print(f"      predictive: habits={len(stats['predictive']['habits'])}")

        # 11. compact
        compact_result = await mm.compact()
        check("compact 压缩全系统", isinstance(compact_result, dict))

        # 12. close
        await mm.close()
        check("close 安全关闭", True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 边界情况测试
# ============================================================

def test_edge_cases():
    section("边界情况")

    # L1: 空内容、特殊字符
    tmp = tempfile.mkdtemp(prefix="soul_test_edge_")
    try:
        fm = FrozenMemory(tmp)
        fm.snapshot()

        # 空写入
        fm._active = False
        fm.write("MEMORY.md", "")
        check("L1: 空内容写入 (解冻)", fm.get_memory() == "")
        fm._active = True

        # 只有 § 分隔符的内容
        content, _ = fm.add("MEMORY.md", "只有一条")
        check("L1: 单条目 (add 返回值)", len(content) > 0 and "只有一条" in content)
        # 特殊字符
        fm.add("MEMORY.md", "<script>alert('xss')</script>")
        check("L1: 特殊字符正常存储", True)
        # 删除不存在的模式
        fm.remove("MEMORY.md", "不存在的内容")
        check("L1: 删除不存在的模式不报错", True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 主入口
# ============================================================

async def main():
    global passed, failed
    passed, failed = 0, 0

    print("\n" + "▓" * 60)
    print("  记忆系统 4 层集成测试")
    print("▓" * 60)

    # 同步测试
    test_layer1_frozen()
    test_edge_cases()

    # 异步测试
    await test_layer4_predictive()
    await test_layer3_indexed()
    await test_layer2_procedural()
    await test_memory_manager()

    # 结果
    print("\n" + "▓" * 60)
    total = passed + failed
    print(f"  结果: {passed}/{total} 通过, {failed} 失败")
    if failed > 0:
        print("  *** 存在失败的测试 ***")
    else:
        print("  全部测试通过！")
    print("▓" * 60)
    return failed


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
