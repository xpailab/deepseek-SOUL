"""Layer 4: 预测记忆 — SOUL 创新。

超越 Hermes 三层记忆的第四层：
- 分析用户行为模式，预测下一步需求
- 基于时间/上下文/历史，主动准备相关信息
- 智能预加载技能和工具
- 检测用户习惯，自动优化工作流

设计原理：
1. 路径预测 — 基于任务历史预测下一步操作
2. 上下文预加载 — 在用户提问前准备相关记忆
3. 习惯学习 — 识别重复模式并提示自动化机会
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class PredictiveMemory:
    """预测记忆 — 第四层（SOUL 创新）。

    三种预测机制：
    1. 任务路径预测 — "你接下来可能要..."
    2. 上下文预加载 — 自动检索相关历史
    3. 习惯检测 — 发现可自动化的重复模式
    """

    def __init__(self, save_path: str = "~/.soul/predictive.json"):
        # 任务路径图: task_type -> [(next_task_type, probability)]
        self._task_graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
        # 用户习惯: habit_name -> {count, last_time, pattern}
        self._habits: dict[str, dict[str, Any]] = {}
        # 时间关联记忆: hour_of_day -> [memory_ids]
        self._temporal_index: dict[int, list[str]] = defaultdict(list)
        # 上下文关联: context_key -> [memory_ids]
        self._context_index: dict[str, list[str]] = defaultdict(list)
        # 配置
        self.min_observations = 3
        self.decay_factor = 0.95
        self.max_predictions = 3
        self._save_path = Path(save_path).expanduser()

    def save(self) -> None:
        """持久化预测数据到 JSON 文件。"""
        import json
        data = {
            "task_graph": {k: list(v) for k, v in self._task_graph.items()},
            "habits": dict(self._habits),
            "temporal": {str(k): v for k, v in self._temporal_index.items()},
            "context": dict(self._context_index),
        }
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        """从 JSON 文件恢复预测数据。"""
        import json
        if not self._save_path.exists():
            return
        try:
            data = json.loads(self._save_path.read_text(encoding="utf-8"))
            self._task_graph = defaultdict(list, {k: [(a, p) for a, p in v] for k, v in data.get("task_graph", {}).items()})
            self._habits = data.get("habits", {})
            self._temporal_index = defaultdict(list, {int(k): v for k, v in data.get("temporal", {}).items()})
            self._context_index = defaultdict(list, data.get("context", {}))
        except Exception:
            pass

    async def observe(
        self,
        current_action: str,
        previous_action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """观察用户行为，更新预测模型。

        Args:
            current_action: 当前执行的操作描述
            previous_action: 上一个操作（用于学习路径）
            context: 当前上下文（时间、项目、目录等）
        """
        now = time.time()
        hour = int(time.localtime(now).tm_hour)

        # 更新任务路径图
        if previous_action:
            self._update_task_graph(previous_action, current_action)

        # 更新时态索引
        self._temporal_index[hour].append(current_action)
        # 限制大小
        if len(self._temporal_index[hour]) > 100:
            self._temporal_index[hour] = self._temporal_index[hour][-50:]

        # 更新上下文索引
        if context:
            ctx_key = self._context_to_key(context)
            self._context_index[ctx_key].append(current_action)
            if len(self._context_index[ctx_key]) > 50:
                self._context_index[ctx_key] = self._context_index[ctx_key][-25:]

        # 更新习惯追踪
        habit_name = self._normalize_action(current_action)
        if habit_name not in self._habits:
            self._habits[habit_name] = {"count": 0, "last_time": now, "pattern": ""}
        self._habits[habit_name]["count"] += 1
        self._habits[habit_name]["last_time"] = now
        if previous_action:
            self._habits[habit_name]["pattern"] = f"通常在 '{previous_action}' 之后"

    async def predict_next_actions(
        self,
        current_action: str,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """预测用户下一步可能需要的操作。

        Returns:
            [{action: str, probability: float, reason: str}, ...]
        """
        predictions: list[dict[str, Any]] = []

        # 1. 路径预测
        path_preds = self._predict_from_path(current_action)
        predictions.extend(path_preds)

        # 2. 时间预测
        now = time.time()
        hour = int(time.localtime(now).tm_hour)
        time_preds = self._predict_from_time(hour, current_action)
        predictions.extend(time_preds)

        # 3. 上下文预测
        if context:
            ctx_preds = self._predict_from_context(context, current_action)
            predictions.extend(ctx_preds)

        # 合并排序，取 top-k
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        # 去重
        seen: set[str] = {current_action}
        unique: list[dict[str, Any]] = []
        for p in predictions:
            if p["action"] not in seen:
                seen.add(p["action"])
                unique.append(p)
                if len(unique) >= self.max_predictions:
                    break

        return unique

    async def detect_habits(self) -> list[dict[str, Any]]:
        """检测用户习惯，返回可自动化的重复模式。

        Returns:
            [{habit: str, frequency: str, suggestion: str}, ...]
        """
        habits: list[dict[str, Any]] = []

        for habit_name, data in self._habits.items():
            count = data["count"]
            if count >= self.min_observations:
                freq = self._describe_frequency(data)
                habits.append({
                    "habit": habit_name,
                    "count": count,
                    "frequency": freq,
                    "pattern": data.get("pattern", ""),
                    "suggestion": self._generate_automation_suggestion(habit_name, data),
                })

        habits.sort(key=lambda x: x["count"], reverse=True)
        return habits

    async def preload_context(
        self,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """预加载相关上下文 — 在用户提问前准备好。

        Returns:
            建议预加载的记忆ID列表
        """
        if not context:
            now = time.time()
            hour = int(time.localtime(now).tm_hour)
            return self._temporal_index.get(hour, [])[-5:]

        ctx_key = self._context_to_key(context)
        return self._context_index.get(ctx_key, [])[-5:]

    async def get_predictive_context_prompt(self) -> str:
        """生成预测上下文的 prompt 片段。

        注入到 system prompt 中，让 Agent 感知到预测。
        """
        habits = await self.detect_habits()
        if not habits:
            return ""

        lines = ["<predictive_context>"]
        lines.append("基于历史模式检测到的用户习惯：")
        for h in habits[:3]:
            lines.append(f"- {h['habit']}: 已重复 {h['count']} 次 ({h['frequency']})")
        lines.append("请根据这些模式主动提供帮助。")
        lines.append("</predictive_context>")

        return "\n".join(lines)

    def _update_task_graph(self, prev: str, curr: str) -> None:
        """更新任务路径图。"""
        prev_key = self._normalize_action(prev)
        curr_key = self._normalize_action(curr)

        entries = self._task_graph[prev_key]

        for i, (action, prob) in enumerate(entries):
            if action == curr_key:
                # 更新概率（指数移动平均）
                new_prob = prob * self.decay_factor + (1 - self.decay_factor)
                entries[i] = (action, new_prob)
                break
        else:
            entries.append((curr_key, 0.3))

        # 归一化概率
        total = sum(p for _, p in entries)
        if total > 0:
            self._task_graph[prev_key] = [(a, p / total) for a, p in entries]

    def _predict_from_path(self, action: str) -> list[dict[str, Any]]:
        """基于路径图的预测。"""
        key = self._normalize_action(action)
        entries = self._task_graph.get(key, [])

        if len(entries) < self.min_observations:
            return []

        entries.sort(key=lambda x: x[1], reverse=True)
        return [
            {"action": a, "probability": round(p, 2), "reason": f"你通常在 '{action}' 之后进行 '{a}'"}
            for a, p in entries[:3]
            if p > 0.2
        ]

    def _predict_from_time(self, hour: int, current: str) -> list[dict[str, Any]]:
        """基于时间的预测。"""
        actions = self._temporal_index.get(hour, [])
        if len(actions) < self.min_observations:
            return []

        # 统计最常见操作
        counter: dict[str, int] = defaultdict(int)
        for a in actions:
            if a != current:
                counter[a] += 1

        total = sum(counter.values())
        if total == 0:
            return []

        return [
            {"action": a, "probability": round(c / total, 2),
             "reason": f"你经常在这个时间段进行 '{a}'"}
            for a, c in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:3]
            if c >= self.min_observations
        ]

    def _predict_from_context(
        self, context: dict[str, Any], current: str
    ) -> list[dict[str, Any]]:
        """基于上下文的预测。"""
        ctx_key = self._context_to_key(context)
        actions = self._context_index.get(ctx_key, [])

        counter: dict[str, int] = defaultdict(int)
        for a in actions:
            if a != current:
                counter[a] += 1

        total = sum(counter.values())
        if total == 0:
            return []

        return [
            {"action": a, "probability": round(c / total, 2),
             "reason": f"在此项目/上下文中你常进行 '{a}'"}
            for a, c in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:3]
            if c >= self.min_observations
        ]

    @staticmethod
    def _normalize_action(action: str) -> str:
        """标准化操作描述。"""
        # 提取核心动词 + 宾语
        action = action.lower().strip()
        # 简单的停用词去除
        for stop in ["请", "帮我", "能不能", "可不可以", "我想", "我要"]:
            action = action.replace(stop, "")
        return action.strip()[:80]

    @staticmethod
    def _context_to_key(context: dict[str, Any]) -> str:
        """上下文转索引键。"""
        parts = []
        if "project" in context:
            parts.append(f"p:{context['project']}")
        if "directory" in context:
            parts.append(f"d:{context['directory']}")
        if "task" in context:
            parts.append(f"t:{context['task']}")
        return "|".join(parts) if parts else "default"

    @staticmethod
    def _describe_frequency(data: dict[str, Any]) -> str:
        """描述频率。"""
        count = data["count"]
        if count >= 50:
            return "非常频繁（几乎每次）"
        elif count >= 20:
            return "频繁（大多数时候）"
        elif count >= 10:
            return "经常"
        elif count >= 5:
            return "偶尔"
        else:
            return "有时"

    @staticmethod
    def _generate_automation_suggestion(
        habit: str, data: dict[str, Any]
    ) -> str:
        """生成自动化建议。"""
        count = data["count"]
        if count >= 20:
            return "建议将此流程设为 cron 定时任务，自动执行"
        elif count >= 10:
            return "建议创建快捷命令或别名简化此操作"
        elif count >= 5:
            return "可以考虑创建技能模板加速此流程"
        else:
            return "继续观察，积累更多数据后给出建议"
