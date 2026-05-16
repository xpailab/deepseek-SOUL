"""LLM 适配器层 — 统一的多提供商接口。

支持 DeepSeek、Claude (Anthropic)、OpenAI 以及兼容 OpenAI API 的任何服务。
所有适配器提供统一的流式和非流式接口。
"""

from soul.llm.base import BaseAdapter, LLMResponse
from soul.llm.registry import AdapterRegistry

__all__ = ["BaseAdapter", "LLMResponse", "AdapterRegistry"]
