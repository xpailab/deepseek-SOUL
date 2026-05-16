"""上下文压缩器 — 当对话历史超出模型窗口时自动触发。

策略：
- 保留关键决策点
- 摘要中间步骤
- 保持工具调用的关键信息
- 递归压缩直到适配窗口
"""

from __future__ import annotations

from typing import Any

from soul.types import Message, MessageRole


class ContextCompressor:
    """对话上下文压缩器。

    支持多种压缩策略，从简单截断到 LLM 摘要。
    """

    def __init__(
        self,
        max_tokens: int = 128000,
        system_token_budget: int = 4000,
        safety_margin: float = 0.9,
    ):
        self.max_tokens = max_tokens
        self.system_budget = system_token_budget
        self.safety_margin = safety_margin
        # 压缩阈值：使用量达到此比例时触发压缩
        self.threshold = 0.8

    def needs_compression(self, messages: list[Message], system_tokens: int = 0) -> bool:
        """检查是否需要压缩。"""
        total = system_tokens + self._estimate_tokens(messages)
        return total > self.max_tokens * self.threshold

    def compress(
        self,
        messages: list[Message],
        system_tokens: int = 0,
        preserve_last: int = 5,
    ) -> list[Message]:
        """压缩消息列表直到适配窗口。

        Args:
            messages: 待压缩的消息列表
            system_tokens: system prompt 已占用的 token 数
            preserve_last: 始终保留最后 N 条消息

        Returns:
            压缩后的消息列表
        """
        if not self.needs_compression(messages, system_tokens):
            return messages

        available = int(self.max_tokens * self.safety_margin) - system_tokens

        # 策略 1: 保留最近的消息 + 关键工具交互
        compressed = self._sliding_window(messages, available, preserve_last)

        # 策略 2: 如果仍然超出，对中间消息做摘要
        if self._estimate_tokens(compressed) > available:
            compressed = self._summarize_middle(compressed, available, preserve_last)

        return compressed

    def compress_to_summary(
        self,
        messages: list[Message],
        max_summary_tokens: int = 2000,
    ) -> str:
        """将消息列表压缩为文本摘要。

        用于需要 LLM 处理的场景，返回提示词用于让 LLM 做摘要。
        """
        summary_parts: list[str] = []
        for msg in messages:
            role = msg.role.value
            content = msg.content[:200] if msg.content else ""
            if msg.tool_calls:
                names = [tc.name for tc in msg.tool_calls]
                content += f" [工具调用: {', '.join(names)}]"
            if msg.tool_results:
                results = []
                for tr in msg.tool_results:
                    status = "成功" if tr.success else "失败"
                    r = str(tr.result)[:100] if tr.result else tr.error or ""
                    results.append(f"{tr.name}({status}): {r}")
                content += f" [结果: {'; '.join(results)}]"
            summary_parts.append(f"[{role}] {content}")

        return "\n".join(summary_parts)

    def _sliding_window(
        self, messages: list[Message], max_tokens: int, preserve_last: int
    ) -> list[Message]:
        """滑动窗口压缩。"""
        if len(messages) <= preserve_last:
            return messages

        preserved = messages[-preserve_last:]
        remaining_budget = max_tokens - self._estimate_tokens(preserved)

        kept: list[Message] = []
        kept_tokens = 0

        for msg in reversed(messages[:-preserve_last]):
            msg_tokens = self._estimate_tokens([msg])
            if kept_tokens + msg_tokens <= remaining_budget:
                kept.insert(0, msg)
                kept_tokens += msg_tokens
            else:
                break

        return kept + preserved

    def _summarize_middle(
        self, messages: list[Message], max_tokens: int, preserve_last: int
    ) -> list[Message]:
        """对中间部分做摘要标记。"""
        if len(messages) <= preserve_last + 2:
            return messages

        preserved = messages[-preserve_last:]
        middle = messages[1:-preserve_last]
        first = messages[:1]

        summary_text = self.compress_to_summary(middle)
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[上下文摘要 — 已压缩 {len(middle)} 条消息]\n{summary_text}",
        )

        return first + [summary_msg] + preserved

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        """粗略估算 token 数。"""
        total = 0
        for msg in messages:
            total += max(1, len(msg.content or "") // 3)
            total += len(msg.tool_calls) * 50
            total += len(msg.tool_results) * 30
        return total

    def create_compression_prompt(self, messages: list[Message]) -> str:
        """生成用于 LLM 压缩的提示词模板。"""
        history_text = self.compress_to_summary(messages)

        return f"""请将以下对话历史压缩为一个简洁的摘要（不超过 500 字）。
保留: 所有决策点、工具调用的关键结果、用户的重要偏好、未完成的任务。

对话历史:
{history_text}

摘要:"""

    async def llm_compress(
        self,
        messages: list[Message],
        adapter: Any,  # BaseAdapter
        max_summary_length: int = 500,
    ) -> str:
        """使用 LLM 进行智能压缩。"""
        prompt = self.create_compression_prompt(messages)
        compressed_msgs = [Message(role=MessageRole.USER, content=prompt)]
        response = await adapter.chat(compressed_msgs)
        return response.content[:max_summary_length]
