"""Atropos RL 环境 — 工具使用强化学习。

将 Agent 工具调用建模为 RL 任务：
- State: 当前对话历史 + 可用工具
- Action: 选择工具 + 参数
- Reward: 正确性 + 效率 + 安全性

适用于 GRPO / PPO / DPO 等 RL 算法的微调数据生成。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AtroposConfig:
    """Atropos RL 环境配置。"""

    # 奖励权重
    reward_correctness: float = 1.0     # 任务完成正确性
    reward_efficiency: float = 0.3      # 工具调用效率（越少越好）
    reward_safety: float = 0.5          # 安全性（不触发护栏）
    reward_penalty_invalid: float = -0.5  # 无效工具调用惩罚
    reward_penalty_timeout: float = -0.3  # 超时惩罚

    # 环境参数
    max_steps: int = 15                 # 每 episode 最大步数
    tool_timeout: int = 120             # 工具超时 (秒)
    parse_timeout: int = 5              # 解析超时 (秒)

    # 输出格式
    output_format: str = "chatml"       # chatml / openai / sharegpt


@dataclass
class AtroposStep:
    """RL 环境中的单步。"""
    state: dict[str, Any]               # 当前对话历史 + 工具列表
    action: dict[str, Any]              # 选择的工具 + 参数
    reward: float                       # 即时奖励
    done: bool                          # episode 是否结束
    info: dict[str, Any]                # 额外信息


class AtroposEnv:
    """Atropos 工具使用 RL 环境。

    将 LLM 的工具调用过程建模为多步决策问题。
    每步 Agent 选择一个工具（或决定不调用工具），环境返回奖励。

    使用示例:
        env = AtroposEnv()
        state = env.reset(task="列出当前目录文件")
        while True:
            action = agent.select_action(state)  # 选择工具 + 参数
            step = env.step(action)
            if step.done:
                break
            agent.learn(step.reward)
    """

    # ── 工具调用解析 ── 11 种模式覆盖主流 LLM 输出格式
    TOOL_CALL_PARSERS: list[tuple[str, str]] = [
        # OpenAI/DeepSeek function calling
        ("function_call", r'<function_call>\s*(\{.*?\})\s*</function_call>'),
        ("tool_call_json", r'"name":\s*"(\w+)"[^}]*"arguments":\s*(\{[^}]+\})'),
        # Anthropic tool_use
        ("anthropic_xml", r'<function_calls>.*?<invoke name="(\w+)">\s*<parameter[^>]*>(.*?)</parameter>'),
        ("anthropic_json", r'{"tool":"(\w+)"[^}]*"input":(\{[^}]+\})'),
        # XML 格式
        ("xml_invoke", r'<invoke\s+name="(\w+)"[^>]*>(.*?)</invoke>'),
        ("xml_tool", r'<tool\s+name="(\w+)">(.*?)</tool>'),
        # Markdown 代码块
        ("markdown_bash", r'```(?:bash|sh|shell)\s*\n(.*?)```'),
        ("markdown_python", r'```(?:python|py)\s*\n(.*?)```'),
        ("markdown_json", r'```(?:json)\s*\n(.*?)```'),
        # 自然语言
        ("nl_action", r'(?:run|execute|运行|执行)\s*(?:command|命令)[：:]\s*(.+)'),
        # 工具名 + 参数
        ("name_args", r'(\w+)\((.*?)\)'),
    ]

    def __init__(self, config: AtroposConfig | None = None):
        self.config = config or AtroposConfig()
        self._history: list[dict[str, Any]] = []
        self._step_count: int = 0
        self._task: str = ""
        self._tools: list[dict[str, Any]] = []
        self._done: bool = False

    # ═══════════════════════════════════════
    # 核心 RL 接口
    # ═══════════════════════════════════════

    def reset(
        self,
        task: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """重置环境，开始新 episode。"""
        self._history = [{"role": "user", "content": task}]
        self._step_count = 0
        self._task = task
        self._tools = tools or []
        self._done = False

        return {
            "messages": self._history.copy(),
            "tools": self._tools,
            "task": task,
        }

    def step(self, action: dict[str, Any]) -> AtroposStep:
        """执行一步工具调用并返回奖励。"""
        self._step_count += 1

        tool_name = action.get("name", action.get("tool", ""))
        tool_args = action.get("arguments", action.get("args", {}))

        # 空动作：Agent 认为任务完成
        if not tool_name or action.get("finish", False):
            self._done = True
            return AtroposStep(
                state=self._get_state(),
                action=action,
                reward=0.0,  # 完成时无额外奖励，由 episode 评估
                done=True,
                info={"reason": "task_complete"},
            )

        # 有效性检查
        if not self._is_valid_tool(tool_name):
            return AtroposStep(
                state=self._get_state(),
                action=action,
                reward=self.config.reward_penalty_invalid,
                done=False,
                info={"reason": "invalid_tool", "tool": tool_name},
            )

        # 超出步数限制
        if self._step_count >= self.config.max_steps:
            self._done = True
            return AtroposStep(
                state=self._get_state(),
                action=action,
                reward=self.config.reward_penalty_invalid,
                done=True,
                info={"reason": "max_steps_exceeded"},
            )

        # 记录历史
        self._history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "name": tool_name,
                "arguments": tool_args,
            }],
        })

        # 即时奖励 = 效率分（少调用多得分）
        efficiency_reward = self.config.reward_efficiency * (1.0 / self._step_count)
        safety_reward = self._check_safety(tool_name, tool_args)

        return AtroposStep(
            state=self._get_state(),
            action=action,
            reward=efficiency_reward + safety_reward,
            done=False,
            info={"step": self._step_count, "tool": tool_name},
        )

    def evaluate_episode(
        self,
        result: str,
        expected_tools: list[str] | None = None,
    ) -> float:
        """episode 结束后评估总分。"""
        total = 0.0

        # 正确性：是否产生了有意义的输出
        if result and len(result) > 10:
            total += self.config.reward_correctness

        # 效率：用到的工具是否必要
        tools_used = [
            m.get("tool_calls", [{}])[0].get("name", "")
            for m in self._history
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if len(tools_used) <= 5:
            total += self.config.reward_efficiency

        # 期望工具匹配
        if expected_tools:
            matched = set(expected_tools) & set(tools_used)
            total += len(matched) / max(len(expected_tools), 1) * self.config.reward_correctness

        return round(total, 3)

    # ═══════════════════════════════════════
    # 工具调用解析
    # ═══════════════════════════════════════

    @classmethod
    def parse_tool_calls(cls, text: str) -> list[dict[str, Any]]:
        """从 LLM 原始输出中解析工具调用 — 11 种格式自动识别。

        Returns:
            [{name: str, arguments: dict, parser: str, confidence: float}, ...]
        """
        results: list[dict[str, Any]] = []

        for parser_name, pattern in cls.TOOL_CALL_PARSERS:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    name = groups[0].strip()
                    args_str = groups[1].strip()
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        arguments = {"raw": args_str}
                    results.append({
                        "name": name,
                        "arguments": arguments,
                        "parser": parser_name,
                        "confidence": 0.9 if parser_name.startswith(("function_call", "tool_call")) else 0.6,
                    })
                elif len(groups) == 1 and parser_name in ("markdown_bash", "nl_action", "name_args"):
                    results.append({
                        "name": "bash",
                        "arguments": {"command": groups[0].strip()},
                        "parser": parser_name,
                        "confidence": 0.5,
                    })

        return results

    @classmethod
    def extract_tool_chain(cls, text: str) -> list[str]:
        """提取工具调用链 — 用于轨迹分析。"""
        calls = cls.parse_tool_calls(text)
        return [c["name"] for c in calls]

    # ═══════════════════════════════════════
    # Trajectory → RL 训练数据
    # ═══════════════════════════════════════

    @classmethod
    def trajectory_to_rl_data(
        cls,
        trajectory: dict[str, Any],
        format: str = "chatml",
    ) -> dict[str, Any]:
        """将执行轨迹转换为 RL 训练数据格式。

        输入: Trajectory (来自 trajectory.py)
        输出: 可用于 GRPO/DPO 微调的配对数据
        """
        steps = trajectory.get("steps", [])
        messages: list[dict[str, str]] = []
        rewards: list[float] = []

        for step in steps:
            role = step.get("role", "user")
            content = step.get("content", "")
            tool_calls = step.get("tool_calls")

            if role == "assistant" and tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {"name": tc["name"], "arguments": tc.get("arguments", {})}
                        for tc in tool_calls
                    ],
                })
                rewards.append(0.1)  # 工具调用小奖励
            elif role == "tool":
                messages.append({
                    "role": "tool",
                    "content": content,
                })
                # 工具成功 = 正奖励，失败 = 负奖励
                is_error = any(
                    e in str(content).lower()
                    for e in ("error", "failed", "denied", "timeout")
                )
                rewards.append(-0.3 if is_error else 0.2)
            else:
                messages.append({"role": role, "content": content})
                rewards.append(0.0)

        return {
            "messages": messages,
            "rewards": rewards,
            "total_reward": round(sum(rewards), 3),
            "format": format,
        }

    # ═══════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════

    def _get_state(self) -> dict[str, Any]:
        return {
            "messages": self._history.copy(),
            "tools": self._tools,
            "remaining_steps": self.config.max_steps - self._step_count,
        }

    def _is_valid_tool(self, name: str) -> bool:
        if not self._tools:
            return True  # 无工具列表时不限制
        valid_names = {t.get("name", "") for t in self._tools}
        valid_names.update({"bash", "file", "web", "browser"})  # 内置工具始终有效
        return name in valid_names

    def _check_safety(self, tool_name: str, args: dict[str, Any]) -> float:
        """安全性检查 — 危险操作扣分。"""
        if tool_name in ("bash", "shell", "exec"):
            command = str(args.get("command", ""))
            dangerous_patterns = [
                r"rm\s+-rf\s+/", r"mkfs\.", r"dd\s+if=",
                r":\(\)\s*\{", r"curl.*\|.*sh", r"sudo\s+rm",
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, command):
                    return self.config.reward_penalty_invalid
        return self.config.reward_safety * 0.1  # 默认安全分
