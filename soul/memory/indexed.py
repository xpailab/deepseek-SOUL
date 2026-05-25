"""Layer 3: FTS5 + LLM 混合检索记忆。

SQLite FTS5 做精确召回（人名/项目名/命令不丢），
LLM 做语义理解和摘要。

设计哲学：不依赖向量数据库，零运维，$5 VPS 即可运行。

中文支持：jieba 预分词 → 空格连接 → unicode61 索引，中英混合搜索。
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import aiosqlite

from soul.types import MemoryEntry, MemoryLayer

# ---------------------------------------------------------------------------
# jieba 延迟加载 — 纯 Python 中文分词，约 500KB，零 C 依赖
# ---------------------------------------------------------------------------
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        try:
            import jieba
            jieba.setLogLevel(20)  # 静默日志
            _jieba = jieba
        except ImportError:
            pass
    return _jieba


# 检测文本是否含 CJK 字符
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


class IndexedMemory:
    """FTS5 + LLM 混合检索 — 第三层。

    表结构:
    - conversations: (id, session_id, role, content, timestamp, metadata)
    - conversation_fts: FTS5 虚拟表，内容为 conversations(content)
    - memory_entries: 结构化记忆条目
    """

    def __init__(self, db_path: str = "~/.soul/memory.db"):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        # LLM 回调 — MemoryManager 注入，IndexedMemory 无需知道 LLM 细节
        self._llm: Callable[[str], Awaitable[str]] | None = None

    def set_llm(self, llm: Callable[[str], Awaitable[str]]) -> None:
        """注入 LLM 回调，启用动态查询扩展和语义重排。"""
        self._llm = llm

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(str(self.db_path))
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            if not self._initialized:
                await self._init_tables()
        return self._db

    async def _init_tables(self) -> None:
        if self._db is None:
            raise RuntimeError("数据库连接未初始化")
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts
                USING fts5(content, role, session_id, tokenize='porter unicode61');

            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                layer TEXT NOT NULL DEFAULT 'indexed',
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                expires_at REAL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversations(session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_conv_timestamp
                ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_mem_importance
                ON memory_entries(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_mem_tags
                ON memory_entries(tags);
        """)
        self._initialized = True

    async def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """存储消息到对话历史。"""
        db = await self._get_db()
        msg_id = f"msg_{int(time.time() * 1000000)}_{secrets.token_hex(4)}"
        ts = time.time()

        # 中文预分词：插入 FTS5 前分词，conversations 表保留原文
        tokenized = self._tokenize(content)

        # 事务包裹，确保 conversations 和 FTS 索引的 rowid 一致
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            """INSERT INTO conversations (id, session_id, role, content, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (msg_id, session_id, role, content, ts, json.dumps(metadata or {})),
        )
        await db.execute(
            "INSERT INTO conversation_fts (content, role, session_id) VALUES (?, ?, ?)",
            (tokenized, role, session_id),
        )
        await db.commit()
        return msg_id

    @staticmethod
    def _tokenize(text: str) -> str:
        """对文本进行中文分词（空格连接），英文保持不变。

        jieba 可用时：中文短语切分为空格分隔的 token，
        unicode61 分词器可正确索引。jieba 不可用时原样返回。

        示例:
            "部署到生产环境" → "部署 到 生产 环境"
            "Python asyncio 并发处理" → "Python asyncio 并发 处理"
            "hello world" → "hello world"  (不变)
        """
        if not _has_cjk(text):
            return text

        jieba = _get_jieba()
        if jieba is None:
            return text  # 降级：jieba 未安装时原样返回

        # jieba.cut 可能返回空白 token，过滤掉
        tokens = [t for t in jieba.cut(text) if t.strip()]
        if not tokens:
            return text
        return " ".join(tokens)

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """清理 FTS5 查询中的特殊字符，防止语法错误。"""
        # 转义 FTS5 特殊字符
        for char in ('"', '(', ')', '*', '^', 'NEAR', 'AND', 'OR', 'NOT'):
            query = query.replace(char, f'"{char}"')
        return query

    async def search_fts(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """FTS5 全文搜索 — 精确召回（中文自动分词）。"""
        db = await self._get_db()
        tokenized = self._tokenize(query)
        sanitized = self._sanitize_fts_query(tokenized)

        try:
            if session_id:
                rows = await db.execute_fetchall(
                    """SELECT c.*, rank FROM conversation_fts f
                       JOIN conversations c ON f.rowid = c.rowid
                       WHERE conversation_fts MATCH ? AND c.session_id = ?
                       ORDER BY rank
                       LIMIT ?""",
                    (sanitized, session_id, limit),
                )
            else:
                rows = await db.execute_fetchall(
                    """SELECT c.*, rank FROM conversation_fts f
                       JOIN conversations c ON f.rowid = c.rowid
                       WHERE conversation_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (sanitized, limit),
                )
        except Exception:
            return []  # FTS5 查询失败时返回空结果

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "rank": row["rank"] if "rank" in row.keys() else 0,
            })
        return results

    async def search_semantic(
        self,
        query: str,
        limit: int = 5,
        use_llm: bool = True,
    ) -> list[dict[str, Any]]:
        """语义搜索 — LLM 动态扩展 + FTS5 精确检索 + LLM 语义重排。

        当 LLM 可用时:
        1. LLM 动态扩展查询词（覆盖同义/相关表述）
        2. FTS5 逐一精确检索
        3. LLM 对结果语义重排

        LLM 不可用时: 退回静态同义词表扩展（向后兼容）。

        Args:
            query: 搜索查询
            limit: 返回结果数上限
            use_llm: 是否启用 LLM 增强（默认 True）
        """
        # 关键词扩展（LLM 或静态）
        if use_llm and self._llm:
            expanded = await self._llm_expand_query(query)
        else:
            expanded = self._expand_query_fallback(query)

        db = await self._get_db()
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for term in expanded:
            tokenized_term = self._tokenize(term)
            try:
                rows = await db.execute_fetchall(
                    """SELECT c.*, rank FROM conversation_fts f
                       JOIN conversations c ON f.rowid = c.rowid
                       WHERE conversation_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (tokenized_term, limit),
                )
                for row in rows:
                    rid = row["id"]
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        results.append({
                            "id": rid,
                            "session_id": row["session_id"],
                            "role": row["role"],
                            "content": row["content"],
                            "timestamp": row["timestamp"],
                            "matched_term": term,
                        })
            except Exception:
                # FTS5 查询语法错误时跳过
                continue

        results.sort(key=lambda x: x["timestamp"], reverse=True)

        # LLM 语义重排（可用时自动启用）
        if use_llm and self._llm and len(results) > limit:
            results = await self._llm_rerank(query, results, top_k=limit)

        return results[:limit]

    async def store_memory_entry(self, entry: MemoryEntry) -> None:
        """存储结构化记忆条目。"""
        db = await self._get_db()
        await db.execute(
            """INSERT OR REPLACE INTO memory_entries
               (id, layer, content, tags, importance, access_count,
                created_at, last_accessed, expires_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.layer.value,
                entry.content,
                json.dumps(entry.tags),
                entry.importance,
                entry.access_count,
                entry.created_at,
                entry.last_accessed,
                entry.expires_at,
                json.dumps(entry.metadata),
            ),
        )
        await db.commit()

    async def retrieve_memories(
        self,
        query: str = "",
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """检索结构化记忆。"""
        db = await self._get_db()
        conditions: list[str] = ["1=1"]
        params: list[Any] = []

        if query:
            conditions.append("content LIKE ?")
            params.append(f"%{query}%")

        if tags:
            tag_conds = " OR ".join(["tags LIKE ?" for _ in tags])
            conditions.append(f"({tag_conds})")
            params.extend([f"%{t}%" for t in tags])

        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        where = " AND ".join(conditions)
        rows = await db.execute_fetchall(
            f"""SELECT * FROM memory_entries
               WHERE {where}
               ORDER BY importance DESC, access_count DESC
               LIMIT ?""",
            (*params, limit),
        )

        entries: list[MemoryEntry] = []
        for row in rows:
            entries.append(MemoryEntry(
                id=row["id"],
                layer=MemoryLayer(row["layer"]),
                content=row["content"],
                tags=json.loads(row["tags"]),
                importance=row["importance"],
                access_count=row["access_count"],
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
                expires_at=row["expires_at"],
                metadata=json.loads(row["metadata"]),
            ))

        # 更新访问计数
        for entry in entries:
            await db.execute(
                "UPDATE memory_entries SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (time.time(), entry.id),
            )
        await db.commit()

        return entries

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
        before: float | None = None,
    ) -> list[dict[str, Any]]:
        """获取会话历史。"""
        db = await self._get_db()
        if before:
            rows = await db.execute_fetchall(
                """SELECT * FROM conversations
                   WHERE session_id = ? AND timestamp < ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, before, limit),
            )
        else:
            rows = await db.execute_fetchall(
                """SELECT * FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, limit),
            )

        results: list[dict[str, Any]] = []
        for row in reversed(rows):
            results.append({
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            })
        return results

    async def compact_old_sessions(self, older_than_days: int = 30) -> int:
        """清理旧会话数据。"""
        db = await self._get_db()
        cutoff = time.time() - older_than_days * 86400

        await db.execute("BEGIN IMMEDIATE")
        rows = await db.execute_fetchall(
            "SELECT rowid FROM conversations WHERE timestamp < ?", (cutoff,)
        )
        rowids = [r[0] for r in rows]

        if rowids:
            for rid in rowids:
                await db.execute(
                    "DELETE FROM conversation_fts WHERE rowid = ?", (rid,)
                )

        cursor = await db.execute(
            "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount

        await db.commit()
        return deleted

    async def _llm_expand_query(self, query: str) -> list[str]:
        """LLM 动态查询扩展 — 生成语义相关的搜索词。

        LLM 可用时：根据查询意图生成 3-5 个同义/相关搜索词，
        覆盖不同表述方式。LLM 不可用时退回静态同义词表。
        """
        if self._llm is None:
            return self._expand_query_fallback(query)

        prompt = f"""你是一个搜索查询扩展助手。给定用户的搜索查询，生成 3-5 个语义相关的搜索词，帮助找到内容不同但含义相关的对话记录。

规则：
- 用「, 」分隔每个搜索词
- 搜索词应是独立的关键词或短语（2-6 字）
- 覆盖同义词、缩写、中英文等价表述
- 不要重复原查询本身
- 只输出搜索词，不要解释

查询: {query}
扩展词:"""

        try:
            response = await asyncio.wait_for(self._llm(prompt), timeout=5.0)
            terms = [t.strip() for t in response.strip().split("，")]
            if not terms:
                terms = [t.strip() for t in response.strip().split(",")]
            terms = [t for t in terms if t and t != query]
            # 原始查询 + LLM 扩展
            return [query] + terms[:5]
        except (TimeoutError, Exception):
            return self._expand_query_fallback(query)

    async def _llm_rerank(
        self, query: str, results: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """LLM 语义重排 — 根据查询意图重新排序搜索结果。

        LLM 不可用或结果 <= 2 条时跳过重排。
        """
        if self._llm is None or len(results) <= 2:
            return results[:top_k]

        # 构建候选项列表
        items = []
        for i, r in enumerate(results[:20]):  # 最多 20 个候选项
            items.append(f"[{i}] {r['role']}: {r['content'][:150]}")

        prompt = f"""你是一个搜索结果排序助手。根据用户查询，对以下对话记录按语义相关性从高到低排序。

查询: {query}

候选项:
{chr(10).join(items)}

请只输出排名最高的 {top_k} 个候选项编号，用逗号分隔（如: 3, 7, 1, 0, 5）。
只输出编号，不要解释。"""

        try:
            response = await asyncio.wait_for(self._llm(prompt), timeout=5.0)
            # 解析编号
            indices = []
            for part in response.strip().replace("，", ",").split(","):
                try:
                    idx = int(part.strip())
                    if 0 <= idx < len(results):
                        indices.append(idx)
                except ValueError:
                    continue
            if indices:
                return [results[i] for i in indices if i < len(results)][:top_k]
        except (TimeoutError, Exception):
            pass

        return results[:top_k]

    def _expand_query_fallback(self, query: str) -> list[str]:
        """静态同义词表 — LLM 不可用时的降级方案。"""
        terms = [query]
        synonyms: dict[str, list[str]] = {
            "部署": ["deploy", "上线", "发布", "release"],
            "错误": ["error", "bug", "失败", "异常", "exception"],
            "数据库": ["database", "db", "sql", "存储"],
            "配置": ["config", "设置", "参数", "settings"],
            "测试": ["test", "testing", "验证", "检查"],
        }
        for kw, syns in synonyms.items():
            if kw in query:
                terms.extend(syns[:3])
        return terms

    async def get_stats(self) -> dict[str, Any]:
        db = await self._get_db()
        conv_count = (await db.execute_fetchall("SELECT COUNT(*) as cnt FROM conversations"))[0]["cnt"]
        mem_count = (await db.execute_fetchall("SELECT COUNT(*) as cnt FROM memory_entries"))[0]["cnt"]
        return {
            "total_conversations": conv_count,
            "total_memories": mem_count,
            "db_path": str(self.db_path),
            "db_size_mb": round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0,
        }

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
