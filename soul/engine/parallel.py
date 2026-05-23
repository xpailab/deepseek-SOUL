"""Parallel Agent — 多智能体并行执行引擎。

一条用户指令同时启动 1-5 个 Agent，各自用不同策略独立执行。
谁先成功就用谁的结果，成功方法自动记录。

使用:
    from soul.engine.parallel import ParallelAgent
    pa = ParallelAgent(agent)
    async for event in pa.execute("帮我打开豆包"):
        print(event)  # {stream_id, type, content, ...}
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from soul.types import Message, MessageRole, StreamChunk


# 不同策略的系统提示
APPROACHES = [
    {
        "name": "直接方案",
        "hint": "分析用户的完整需求，用最直接的方法完成全部步骤。不要只做第一步就停止。任务包含多步时必须全部执行完。",
    },
    {
        "name": "分步方案",
        "hint": "把完整任务拆成小步骤，逐步执行。每步完成后检查是否还有未完成的步骤，直到全部完成。",
    },
    {
        "name": "搜索方案",
        "hint": "先搜索相关信息或已有解决方案，再执行全部步骤。确保完成任务的所有部分。",
    },
    {
        "name": "替代方案",
        "hint": "用不同工具或思路完成全部任务。如常规方法不行，尝试脚本等替代方案。必须完成所有步骤。",
    },
    {
        "name": "快速方案",
        "hint": "快速完成全部任务。跳过不必要的检查，但必须完成用户要求的每一个操作。",
    },
]


CLASSIFY_PROMPT = """判断这条消息是闲聊还是任务。返回 JSON: {"difficulty":1-5, "is_chat":true/false, "tools_needed":true/false}

is_chat=true: 问候、寒暄、闲聊、询问你是谁、表示感谢或再见。不需要执行任何操作。
is_chat=false: 要求执行具体操作。包括打开程序、搜索、创建文件、运行命令、读写数据等。

difficulty:
1 = 纯闲聊，不需操作
2 = 单一操作（打开XX、搜索XX、读取XX）
3 = 包含2个操作或用了连接词（并/然后/接着/同时）
4-5 = 多步骤复杂任务（创建+配置+部署等）

只返回 JSON。

消息: """


async def _llm_classify(task: str, adapter) -> int:
    """用 LLM 判断任务复杂度，失败时回退到简单估算。"""

    # 快速关键词检查（秒级，不调 LLM 先）
    action_keywords = ["打开", "搜索", "创建", "发送", "运行", "删除", "下载", "部署",
                       "安装", "配置", "转换", "修改", "写入", "读取", "启动", "关闭"]
    has_action = any(w in task for w in action_keywords)
    has_connector = any(w in task for w in ["然后", "接着", "并且", "同时", "之后", "再"])

    try:
        msgs = [Message(role=MessageRole.USER, content=CLASSIFY_PROMPT + task)]
        resp = await adapter.chat(msgs, max_tokens=80, temperature=0)
        content = resp.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            import json
            data = json.loads(content[start:end])
            is_chat = data.get("is_chat", False)
            # 如果 LLM 说 is_chat 但消息明显是操作指令 → 覆写
            if is_chat and has_action:
                action_count = sum(1 for w in action_keywords if w in task)
                score = 2 + (action_count - 1) + (1 if has_connector else 0)
                if len(task) > 30: score += 1
                return min(score, 5)
            if is_chat or not data.get("tools_needed"):
                return 1
            return max(1, min(data.get("difficulty", 2), 5))
    except Exception:
        pass

    # 回退：关键词匹配
    if not has_action:
        return 1
    # 基础分=2，每多一个动作词+1，每多一个连接词+1
    action_count = sum(1 for w in action_keywords if w in task)
    score = 2 + (action_count - 1) + (1 if has_connector else 0)
    if len(task) > 30:
        score += 1
    return min(score, 5)


# 操作型动词 — 任务目标是产生结果而非获取信息
_ACTION_VERBS = [
    "打开", "关闭", "启动", "停止", "创建", "新建", "删除", "移除",
    "安装", "卸载", "部署", "发布", "推送", "提交", "保存", "写入",
    "重命名", "移动", "复制", "剪切", "压缩", "解压", "下载", "上传",
    "发送", "配置", "设置", "修改", "更新", "升级", "回滚",
    "open", "close", "start", "stop", "create", "new", "delete", "remove",
    "install", "uninstall", "deploy", "publish", "push", "commit", "save", "write",
    "rename", "move", "copy", "cut", "zip", "unzip", "download", "upload",
    "send", "config", "set", "modify", "update", "upgrade",
]

# 探索型动词 — 任务目标是获取信息/分析（多方案结果有互补价值）
_EXPLORE_VERBS = [
    "分析", "评估", "审查", "检查", "查找", "搜索", "查询", "统计",
    "总结", "比较", "对比", "解释", "说明", "理解", "研究", "探索",
    "诊断", "排查", "调试", "监控", "预测", "生成报告", "整理",
    "报错", "错误原因", "为什么", "怎么回事", "崩了", "挂了",
    "analyze", "review", "audit", "check", "find", "search", "query",
    "summarize", "compare", "explain", "understand", "research", "explore",
    "diagnose", "debug", "monitor", "predict", "report", "error",
]


class ParallelAgent:
    """多智能体并行执行器。

    根据任务难度自动决定启动 1-5 个 Agent，各自用不同策略独立执行。

    竞速模式:
    - 操作型任务 (创建/删除/部署/安装...): 任一方案成功 → 立即取消其余方案
    - 探索型任务 (分析/评估/搜索/诊断...): 收集所有方案结果后合并最优
    """

    def __init__(self, agent_factory):
        """
        Args:
            agent_factory: 创建 Agent 实例的工厂函数 async () -> Agent
        """
        self._factory = agent_factory
        self._approaches = APPROACHES
        self._cancel: asyncio.Event | None = None

    @staticmethod
    def _is_action_task(task: str) -> bool:
        """判断是否为操作型任务（竞速取消 vs 收集全部结果）。

        操作型: 目标产生具体结果 (创建/删除/部署/安装)，任一成功即完成
        探索型: 目标获取信息/分析 (查错/对比/评估)，多方案互补
        """
        task_lower = task.lower()
        action_score = sum(1 for w in _ACTION_VERBS if w.lower() in task_lower)
        explore_score = sum(1 for w in _EXPLORE_VERBS if w.lower() in task_lower)
        if explore_score > action_score:
            return False
        # 平局: 有明确操作动词 → 竞速；无明显动词 → 收集 (安全默认)
        if action_score == 0 and explore_score == 0:
            return False
        return True

    async def execute(
        self,
        user_message: str,
        session_id: str = "",
        agent_count: int = 2,
        race_mode: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """并行执行任务，返回事件流。

        Args:
            user_message: 用户消息
            session_id: 会话 ID
            agent_count: 并行 Agent 数量 (1-5)
            race_mode: True=竞速(谁成功谁赢立即取消其余),
                       False=收集全部结果,
                       None=自动判断(操作型→竞速, 探索型→收集)
        """
        agent_count = max(1, min(agent_count, 5))
        approaches = self._approaches[:agent_count]

        # 自动判断竞速模式
        if race_mode is None:
            race_mode = self._is_action_task(user_message)

        # 竞速模式用取消事件
        self._cancel = asyncio.Event() if race_mode else None

        yield {
            "stream_id": "_meta",
            "type": "start",
            "count": agent_count,
            "approaches": [a["name"] for a in approaches],
            "task": user_message,
            "race_mode": race_mode,
        }

        # 并行启动所有 Agent
        tasks = []
        for i, approach in enumerate(approaches):
            sid = f"{session_id}_p{i}" if session_id else f"parallel_{i}"
            tasks.append(self._run_one(
                i, approach, user_message, sid,
                cancel_event=self._cancel,
            ))

        results: dict[int, Any] = {}
        first_success_stream = None

        async for stream_id, event in _merge_streams(tasks):
            if event["type"] == "done":
                results[stream_id] = event
                if event.get("success") and first_success_stream is None:
                    first_success_stream = stream_id
                    # 竞速模式：第一个成功即触发取消
                    if self._cancel:
                        self._cancel.set()
            yield event

        # 选最佳结果
        if first_success_stream is not None:
            winner = first_success_stream
        elif results:
            winner = max(results.keys(), key=lambda k: results[k].get("tools_ok", 0))
        else:
            winner = 0

        best = results.get(winner, {"content": "所有方案均未能完成任务。"})
        stopping = "race" if race_mode else "merged"

        yield {
            "stream_id": "_meta",
            "type": "finished",
            "winner": winner,
            "content": best.get("content", ""),
            "winner_name": approaches[winner]["name"] if winner < len(approaches) else "未知",
            "total_agents": agent_count,
            "completed_agents": len(results),
            "stopping": stopping,
            "results": {
                str(k): {
                    "success": v.get("success", False),
                    "tools": v.get("tools_called", 0),
                    "tools_ok": v.get("tools_ok", 0),
                }
                for k, v in results.items()
            },
        }

    async def _run_one(
        self, stream_id: int, approach: dict, user_message: str, session_id: str,
        cancel_event: asyncio.Event | None = None,
    ):
        """运行单个 Agent，产出事件流。cancel_event 触发时提前终止。"""
        agent = None
        cancelled = False
        try:
            agent = await self._factory()
            await agent.initialize()

            hint = approach["hint"]
            memory_ctx = await agent.memory.query_for_prompt(user_message)
            base_prompt = agent.prompt_builder.build_system_prompt(
                tools=agent.tools.to_api_schemas(),
                extra_context=memory_ctx,
            )
            full_system_prompt = (
                base_prompt
                + f"\n\n## 执行策略: {approach['name']}\n{hint}\n"
                + "重要: 用户的任务可能包含多个步骤。必须完成所有步骤后才能停止。"
                + "如果任务包含'并/然后/接着/同时'等连接词，说明有多个操作需要执行。"
                + "每完成一步，检查是否还有未完成的步骤，直到全部完成。"
            )
            tools_ok = 0
            tools_called = 0
            content = ""

            yield stream_id, {
                "stream_id": str(stream_id),
                "type": "agent_start",
                "approach": approach["name"],
                "hint": hint,
            }

            async for chunk in agent.chat_stream(
                user_message, session_id=session_id, system_prompt=full_system_prompt
            ):
                # 竞速取消检查：每个 chunk 后检查是否已有方案成功
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break

                evt: dict[str, Any] = {"stream_id": str(stream_id)}

                if chunk.content:
                    evt["type"] = "content"
                    evt["content"] = chunk.content
                    content += chunk.content
                if chunk.tool_call:
                    evt["type"] = "tool"
                    evt["tool"] = chunk.tool_call.name
                    tools_called += 1
                if chunk.tool_result:
                    evt["type"] = "result"
                    evt["tool"] = chunk.tool_result.name
                    evt["success"] = chunk.tool_result.success
                    if chunk.tool_result.success:
                        evt["text"] = str(chunk.tool_result.result)[:200]
                        tools_ok += 1
                    else:
                        evt["text"] = chunk.tool_result.error or "失败"
                if chunk.finish_reason:
                    evt["finish"] = chunk.finish_reason

                if evt.get("type"):
                    yield stream_id, evt

            # 该 Agent 完成（或被取消）
            yield stream_id, {
                "stream_id": str(stream_id),
                "type": "done",
                "success": (tools_ok > 0 or len(content) > 30) and not cancelled,
                "content": "[已取消 — 其他方案已成功]" if cancelled else content,
                "tools_called": tools_called,
                "tools_ok": tools_ok,
                "approach": approach["name"],
                "cancelled": cancelled,
            }

        except Exception as e:
            if cancelled:
                return
            yield stream_id, {
                "stream_id": str(stream_id),
                "type": "done",
                "success": False,
                "content": f"错误: {e}",
                "approach": approach["name"],
            }
        finally:
            if agent:
                try:
                    await agent.shutdown()
                except Exception:
                    pass


async def _merge_streams(tasks):
    """合并多个 Agent 的事件流，按到达顺序输出。"""
    queue: asyncio.Queue = asyncio.Queue()
    active = len(tasks)

    async def wrap(coro):
        nonlocal active
        async for item in coro:
            await queue.put(item)
        active -= 1
        await queue.put(None)  # 结束标记

    wrappers = [asyncio.create_task(wrap(t)) for t in tasks]

    finished = 0
    while finished < len(tasks):
        item = await queue.get()
        if item is None:
            finished += 1
        else:
            yield item

    for w in wrappers:
        w.cancel()
