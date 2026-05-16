"""Layer 3: FTS5 + LLM 混合检索记忆。

SQLite FTS5 做精确召回（人名/项目名/命令不丢），
LLM 做语义理解和摘要。

设计哲学：不依赖向量数据库，零运维，$5 VPS 即可运行。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

from soul.types import MemoryEntry, MemoryLayer, MessageRole


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
        db = self._db
        if db is None:
            return
        await db.executescript("""
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
        msg_id = f"msg_{int(time.time() * 1000)}_{hash(content) % 10000}"
        ts = time.time()

        await db.execute(
            """INSERT INTO conversations (id, session_id, role, content, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (msg_id, session_id, role, content, ts, json.dumps(metadata or {})),
        )
        # 同步到 FTS 索引
        await db.execute(
            "INSERT INTO conversation_fts (content, role, session_id) VALUES (?, ?, ?)",
            (content, role, session_id),
        )
        await db.commit()
        return msg_id

    async def search_fts(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """FTS5 全文搜索 — 精确召回。

        Args:
            query: 搜索关键词
            session_id: 限制搜索范围（可选）
            limit: 返回条数

        Returns:
            匹配的对话记录列表
        """
        db = await self._get_db()

        if session_id:
            rows = await db.execute_fetchall(
                """SELECT c.*, rank FROM conversation_fts f
                   JOIN conversations c ON f.rowid = c.rowid
                   WHERE conversation_fts MATCH ? AND c.session_id = ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, session_id, limit),
            )
        else:
            rows = await db.execute_fetchall(
                """SELECT c.*, rank FROM conversation_fts f
                   JOIN conversations c ON f.rowid = c.rowid
                   WHERE conversation_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            )

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
    ) -> list[dict[str, Any]]:
        """语义搜索 — 使用关键词扩展进行更广的匹配。

        不依赖向量数据库，而是用 FTS5 的模糊匹配 +
        关键词扩展实现近似的语义搜索。
        """
        # 关键词扩展
        expanded = self._expand_query(query)

        db = await self._get_db()
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for term in expanded:
            try:
                rows = await db.execute_fetchall(
                    """SELECT c.*, rank FROM conversation_fts f
                       JOIN conversations c ON f.rowid = c.rowid
                       WHERE conversation_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (term, limit),
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

        cursor = await db.execute(
            "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount

        # 清理 FTS 索引
        await db.execute("INSERT INTO conversation_fts(conversation_fts) VALUES('optimize')")
        await db.commit()
        return deleted

    def _expand_query(self, query: str) -> list[str]:
        """扩展搜索查询词。"""
        terms = [query]
        # 简单的同义词/相关词扩展
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
