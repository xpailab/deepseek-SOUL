"""用户建模 — Honcho 风格的辩证推理 + Multi-Peer 架构。

核心机制:
1. 辩证推理 (Dialectical Reasoning): 多Pass分析用户行为
   - Pass 0: 原始观察 → 收集对话模式、工具使用、反馈信号
   - Pass 1: 模式提取 → 识别偏好、技能水平、工作流习惯
   - Pass 2: 综合推断 → 生成个性化建议、预测需求

2. Multi-Peer 架构: 同一用户可有多个 AI Peer
   - coding peer: 代码开发视角
   - writing peer: 文档写作视角
   - devops peer: 基础设施视角
   - data peer: 数据分析视角
   每个 Peer 维护独立的交互风格和用户画像子集

3. 与 FrozenMemory 集成: USER.md 作为持久化层，UserModel 作为智能层
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════

class PeerRole(str, Enum):
    """AI Peer 角色。"""
    CODING = "coding"
    WRITING = "writing"
    DEVOPS = "devops"
    DATA = "data"
    GENERAL = "general"


class SkillLevel(str, Enum):
    UNKNOWN = "unknown"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class TechnologyProfile:
    """用户在特定技术领域的画像。"""
    name: str                           # 技术名称 (Python, Docker, React...)
    level: SkillLevel = SkillLevel.UNKNOWN
    usage_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    success_rate: float = 1.0           # 该技术相关任务的成功率
    common_errors: list[str] = field(default_factory=list)


@dataclass
class PeerConfig:
    """单个 Peer 的配置。"""
    role: PeerRole
    name: str
    description: str
    # 交互风格
    communication_style: str = "concise"    # concise / detailed / socratic
    expertise_domains: list[str] = field(default_factory=list)
    # 该 Peer 关注的用户画像维度
    track_preferences: bool = True
    track_skills: bool = True
    track_patterns: bool = True
    # 建议生成
    auto_suggest: bool = True


@dataclass
class DialecticSnapshot:
    """辩证推理快照 — 一次多Pass分析的结果。"""
    timestamp: float
    trigger: str                        # 触发原因
    # Pass 0: 原始观察
    observations: list[str] = field(default_factory=list)
    # Pass 1: 模式提取
    patterns: list[str] = field(default_factory=list)
    # Pass 2: 综合推断
    insights: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.5             # 综合置信度


@dataclass
class UserProfileData:
    """用户画像完整数据。"""
    # 基础信息
    preferred_language: str = ""         # zh/en/ja...
    communication_style: str = ""        # 沟通偏好
    active_hours: list[int] = field(default_factory=list)  # 活跃时段 (小时)

    # 技术能力
    technologies: dict[str, TechnologyProfile] = field(default_factory=dict)

    # 行为模式
    common_workflows: list[str] = field(default_factory=list)  # 常见工作流
    frequent_commands: list[str] = field(default_factory=list)  # 常用命令
    tool_preferences: dict[str, float] = field(default_factory=dict)  # 工具偏好 + 频率

    # 推断目标
    inferred_goals: list[str] = field(default_factory=list)
    current_project: str = ""

    # 辩证推理历史
    dialectic_snapshots: list[DialecticSnapshot] = field(default_factory=list)

    # 元数据
    first_interaction: float = 0.0
    last_interaction: float = 0.0
    total_interactions: int = 0
    version: int = 1


# ═══════════════════════════════════════════
# Multi-Peer 管理器
# ═══════════════════════════════════════════

DEFAULT_PEERS: dict[PeerRole, PeerConfig] = {
    PeerRole.CODING: PeerConfig(
        role=PeerRole.CODING,
        name="编码助手",
        description="专注于代码开发、调试、重构、架构设计",
        communication_style="concise",
        expertise_domains=["python", "javascript", "typescript", "go", "rust", "react", "fastapi", "docker"],
    ),
    PeerRole.WRITING: PeerConfig(
        role=PeerRole.WRITING,
        name="写作助手",
        description="专注于技术文档、方案设计、报告撰写",
        communication_style="detailed",
        expertise_domains=["markdown", "latex", "documentation", "proposal", "report"],
    ),
    PeerRole.DEVOPS: PeerConfig(
        role=PeerRole.DEVOPS,
        name="运维助手",
        description="专注于部署、监控、CI/CD、基础设施",
        communication_style="concise",
        expertise_domains=["docker", "kubernetes", "terraform", "ansible", "github_actions", "prometheus"],
    ),
    PeerRole.DATA: PeerConfig(
        role=PeerRole.DATA,
        name="数据助手",
        description="专注于数据分析、可视化、机器学习",
        communication_style="socratic",
        expertise_domains=["pandas", "numpy", "matplotlib", "sql", "machine_learning", "statistics"],
    ),
    PeerRole.GENERAL: PeerConfig(
        role=PeerRole.GENERAL,
        name="通用助手",
        description="通用任务处理",
        communication_style="concise",
        expertise_domains=[],
    ),
}


class MultiPeerManager:
    """Multi-Peer 管理器 — 同一用户的不同 AI Peer。

    每个 Peer:
    - 有独立的交互风格和专长领域
    - 共享用户基础画像，但有各自的关注维度
    - 根据上下文自动选择或用户手动切换
    """

    def __init__(self):
        self._peers: dict[PeerRole, PeerConfig] = dict(DEFAULT_PEERS)
        self._active_peer: PeerRole = PeerRole.GENERAL

    @property
    def active_peer(self) -> PeerConfig:
        return self._peers[self._active_peer]

    def switch(self, role: PeerRole) -> PeerConfig:
        """切换当前 Peer。"""
        self._active_peer = role
        return self._peers[role]

    def auto_select(self, task: str) -> PeerConfig:
        """根据任务内容自动选择最合适的 Peer。"""
        scores: dict[PeerRole, int] = {}

        task_lower = task.lower()
        for role, peer in self._peers.items():
            score = 0
            for domain in peer.expertise_domains:
                if domain.lower() in task_lower:
                    score += 10
                # 部分匹配
                parts = domain.split("_")
                for p in parts:
                    if len(p) > 2 and p in task_lower:
                        score += 3
            scores[role] = score

        # 如果有明显匹配，选最高分；否则保持 GENERAL
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 5:
            self._active_peer = best
        else:
            self._active_peer = PeerRole.GENERAL

        return self._peers[self._active_peer]

    def list_peers(self) -> list[dict[str, Any]]:
        return [
            {
                "role": role.value,
                "name": peer.name,
                "description": peer.description,
                "style": peer.communication_style,
                "domains": peer.expertise_domains,
                "active": role == self._active_peer,
            }
            for role, peer in self._peers.items()
        ]

    def get_peer_prompt_fragment(self) -> str:
        """生成当前 Peer 的 prompt 注入片段。"""
        peer = self.active_peer
        lines = [f"<peer role=\"{peer.role.value}\" style=\"{peer.communication_style}\">"]
        lines.append(f"  你当前以 {peer.name} 身份工作。")
        if peer.expertise_domains:
            lines.append(f"  专长领域: {', '.join(peer.expertise_domains)}")
        lines.append("</peer>")
        return "\n".join(lines)


# ═══════════════════════════════════════════
# 辩证推理分析器
# ═══════════════════════════════════════════

class DialecticAnalyzer:
    """辩证推理分析器 — 多Pass分析用户行为。

    Pass 0 (观察): 收集原始交互数据
    Pass 1 (模式): 从观察中提取可重复模式
    Pass 2 (综合): 生成洞察和建议

    使用示例:
        analyzer = DialecticAnalyzer()
        snapshot = analyzer.analyze(
            recent_messages=[...],
            user_profile=profile,
        )
        # snapshot.insights → ["用户开始学习 Rust", "部署频率从周2次增加到日1次"]
    """

    # ── 技术关键词映射 ──
    TECH_KEYWORDS: dict[str, list[str]] = {
        "python": ["python", "django", "flask", "fastapi", "pytest", "pip", "venv"],
        "javascript": ["javascript", "node", "react", "vue", "nextjs", "npm", "pnpm"],
        "go": ["golang", "go mod", "goroutine", "go build"],
        "rust": ["rust", "cargo", "rustc", "crate", "tokio"],
        "docker": ["docker", "dockerfile", "compose", "k8s", "kubernetes"],
        "database": ["sql", "postgresql", "mysql", "mongodb", "redis", "sqlite"],
        "devops": ["deploy", "ci/cd", "github actions", "jenkins", "terraform"],
        "data": ["pandas", "numpy", "matplotlib", "ml", "training", "model"],
    }

    def __init__(self):
        self._observation_buffer: list[dict[str, Any]] = []
        self._last_analysis_time: float = 0.0
        self._min_observations: int = 5       # 至少收集 5 条才开始分析
        self._analysis_interval: int = 3600    # 每小时的交互数据触发一次分析
        self._max_snapshots: int = 20          # 保留最近 20 条快照

    def observe(
        self,
        message: str,
        role: str = "user",
        tools_used: list[str] | None = None,
        success: bool = True,
    ) -> None:
        """记录一次交互观察。"""
        self._observation_buffer.append({
            "timestamp": time.time(),
            "role": role,
            "content": message[:300],
            "tools": tools_used or [],
            "success": success,
        })
        # 限制缓冲区大小
        if len(self._observation_buffer) > 500:
            self._observation_buffer = self._observation_buffer[-300:]

    def analyze(
        self,
        user_profile: UserProfileData,
        force: bool = False,
    ) -> DialecticSnapshot | None:
        """执行辩证分析 — 如果积累足够观察或强制触发。"""
        if not force:
            if len(self._observation_buffer) < self._min_observations:
                return None
            if time.time() - self._last_analysis_time < self._analysis_interval:
                return None

        self._last_analysis_time = time.time()
        recent = self._observation_buffer[-50:]  # 分析最近 50 条

        # ── Pass 0: 原始观察 ──
        observations = self._pass0_observe(recent)

        # ── Pass 1: 模式提取 ──
        patterns = self._pass1_extract_patterns(recent, user_profile)

        # ── Pass 2: 综合推断 ──
        insights, recommendations = self._pass2_synthesize(
            observations, patterns, user_profile
        )

        confidence = self._calculate_confidence(observations, patterns, insights)

        snapshot = DialecticSnapshot(
            timestamp=time.time(),
            trigger="auto" if not force else "manual",
            observations=observations,
            patterns=patterns,
            insights=insights,
            recommendations=recommendations,
            confidence=confidence,
        )

        # 存到用户画像
        user_profile.dialectic_snapshots.append(snapshot)
        if len(user_profile.dialectic_snapshots) > self._max_snapshots:
            user_profile.dialectic_snapshots = user_profile.dialectic_snapshots[-self._max_snapshots:]

        return snapshot

    # ── Pass 0: 原始观察 ──
    def _pass0_observe(self, recent: list[dict[str, Any]]) -> list[str]:
        """收集原始观察事实。"""
        obs: list[str] = []

        # 技术栈使用
        tech_hits: dict[str, int] = {}
        for item in recent:
            content = item.get("content", "").lower()
            for tech, keywords in self.TECH_KEYWORDS.items():
                for kw in keywords:
                    if kw in content:
                        tech_hits[tech] = tech_hits.get(tech, 0) + 1

        for tech, count in sorted(tech_hits.items(), key=lambda x: x[1], reverse=True)[:5]:
            if count >= 2:
                obs.append(f"使用 {tech} 相关 ({count} 次)")

        # 工具使用频率
        tool_counts: dict[str, int] = {}
        for item in recent:
            for t in item.get("tools", []):
                tool_counts[t] = tool_counts.get(t, 0) + 1

        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            obs.append(f"使用工具 {tool} ({count} 次)")

        # 成功率
        successes = sum(1 for item in recent if item.get("success"))
        if len(recent) > 0:
            obs.append(f"任务成功率: {successes}/{len(recent)} ({int(successes/len(recent)*100)}%)")

        return obs

    # ── Pass 1: 模式提取 ──
    def _pass1_extract_patterns(
        self, recent: list[dict[str, Any]], profile: UserProfileData
    ) -> list[str]:
        """从观察中提取可重复模式。"""
        patterns: list[str] = []

        # 工作流模式检测
        user_msgs = [item for item in recent if item.get("role") == "user"]
        if len(user_msgs) >= 3:
            # 检测重复任务类型
            task_types: dict[str, int] = {}
            for msg in user_msgs[-20:]:
                content = msg.get("content", "")
                if "调试" in content or "debug" in content.lower():
                    task_types["debug"] = task_types.get("debug", 0) + 1
                if "部署" in content or "deploy" in content.lower():
                    task_types["deploy"] = task_types.get("deploy", 0) + 1
                if "测试" in content or "test" in content.lower():
                    task_types["test"] = task_types.get("test", 0) + 1
                if "重构" in content or "refactor" in content.lower():
                    task_types["refactor"] = task_types.get("refactor", 0) + 1
                if "文档" in content or "doc" in content.lower():
                    task_types["doc"] = task_types.get("doc", 0) + 1

            for task_type, count in task_types.items():
                if count >= 3:
                    patterns.append(f"高频任务类型: {task_type} (最近{len(user_msgs)}条中 {count} 次)")

        # 时间模式
        hours: list[int] = []
        for item in recent:
            ts = item.get("timestamp", 0)
            if ts > 0:
                hours.append(int(time.localtime(ts).tm_hour))

        if hours:
            # 找出最活跃的时段
            hour_counts: dict[int, int] = {}
            for h in hours:
                hour_counts[h] = hour_counts.get(h, 0) + 1
            peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            peak_str = ", ".join(f"{h}时" for h, _ in peak_hours)
            patterns.append(f"活跃时段: {peak_str}")

            # 更新画像
            profile.active_hours = [h for h, _ in peak_hours]

        return patterns

    # ── Pass 2: 综合推断 ──
    def _pass2_synthesize(
        self,
        observations: list[str],
        patterns: list[str],
        profile: UserProfileData,
    ) -> tuple[list[str], list[str]]:
        """综合 Pass 0 和 Pass 1 的发现，生成洞察和建议。"""
        insights: list[str] = []
        recommendations: list[str] = []

        # 技术学习信号
        for obs in observations:
            if "使用" in obs and "相关" in obs:
                tech = obs.split(" 使用 ")[1].split(" 相关")[0]
                if tech not in profile.technologies:
                    insights.append(f"用户开始接触 {tech}")
                    recommendations.append(f"建议为 {tech} 准备入门资源和技能模板")
                elif profile.technologies[tech].level == SkillLevel.BEGINNER:
                    insights.append(f"用户持续使用 {tech}，技能在提升中")
                    recommendations.append(f"可以逐步为 {tech} 推荐更高级的模式")

        # 工作流优化建议
        for pattern in patterns:
            if "高频任务类型" in pattern:
                task_type = pattern.split(": ")[1].split(" ")[0]
                recommendations.append(f"检测到高频 {task_type} 任务，建议创建快捷命令或技能模板")
            if "活跃时段" in pattern:
                recommendations.append("可根据活跃时段预加载相关上下文和工具")

        # 通用建议
        if profile.total_interactions >= 10 and not profile.common_workflows:
            insights.append("用户已建立稳定的交互模式")
            recommendations.append("建议生成个人快捷命令别名")

        return insights, recommendations

    def _calculate_confidence(
        self,
        observations: list[str],
        patterns: list[str],
        insights: list[str],
    ) -> float:
        """计算分析置信度。"""
        score = 0.3  # 基础分
        score += min(len(observations) * 0.1, 0.3)  # 观察多 → 置信度高
        score += min(len(patterns) * 0.1, 0.2)       # 模式多 → 置信度高
        score += min(len(insights) * 0.05, 0.2)      # 洞察多 → 置信度高
        return round(min(score, 1.0), 2)


# ═══════════════════════════════════════════
# 用户模型 — 统一管理
# ═══════════════════════════════════════════

class UserModel:
    """用户模型 — Honcho 风格辩证推理 + Multi-Peer。

    使用:
        model = UserModel()
        model.observe_message("帮我部署到 k8s", tools=["bash"], success=True)
        model.select_peer_for_task("部署到 k8s")
        insights = model.analyze()
    """

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.profile = UserProfileData()
        self.peers = MultiPeerManager()
        self.analyzer = DialecticAnalyzer()
        self._profile_path = self.workspace / "user_profile.json"
        self._load()

    # ═══════════════════════ 观察 API ═══════════════════════

    def observe_message(
        self,
        content: str,
        role: str = "user",
        tools_used: list[str] | None = None,
        success: bool = True,
    ) -> None:
        """记录一次消息观察。"""
        now = time.time()

        # 更新用户画像
        if self.profile.first_interaction == 0:
            self.profile.first_interaction = now
        self.profile.last_interaction = now
        self.profile.total_interactions += 1

        # 更新技术能力追踪
        content_lower = content.lower()
        for tech, keywords in self.analyzer.TECH_KEYWORDS.items():
            for kw in keywords:
                if kw in content_lower:
                    if tech not in self.profile.technologies:
                        self.profile.technologies[tech] = TechnologyProfile(
                            name=tech,
                            level=SkillLevel.BEGINNER,
                            first_seen=now,
                        )
                    tp = self.profile.technologies[tech]
                    tp.usage_count += 1
                    tp.last_seen = now
                    if not success:
                        tp.success_rate = (tp.success_rate * (tp.usage_count - 1) + 0) / tp.usage_count
                    break

        # 更新工具偏好
        for tool in (tools_used or []):
            self.profile.tool_preferences[tool] = self.profile.tool_preferences.get(tool, 0) + 1

        # 辩证分析器观察
        self.analyzer.observe(content, role, tools_used, success)

        # 自动选择 Peer
        if role == "user" and success:
            self.peers.auto_select(content)

    # ═══════════════════════ 分析 API ═══════════════════════

    def analyze(self, force: bool = False) -> DialecticSnapshot | None:
        """运行辩证分析 — 需要足够积累或手动触发。"""
        return self.analyzer.analyze(self.profile, force=force)

    def get_latest_insights(self) -> list[str]:
        """获取最近一次分析的洞察。"""
        if self.profile.dialectic_snapshots:
            return self.profile.dialectic_snapshots[-1].insights
        return []

    def get_recommendations(self) -> list[str]:
        """获取最近一次分析的建议。"""
        if self.profile.dialectic_snapshots:
            return self.profile.dialectic_snapshots[-1].recommendations
        return []

    # ═══════════════════════ Peer API ═══════════════════════

    def select_peer_for_task(self, task: str) -> PeerConfig:
        """根据任务选择最佳 Peer 并返回。"""
        return self.peers.auto_select(task)

    def switch_peer(self, role: str) -> PeerConfig:  # noqa: F811 (intentional re-definition for user-facing API)
        """手动切换 Peer。"""
        try:
            peer_role = PeerRole(role)
        except ValueError:
            peer_role = PeerRole.GENERAL
        return self.peers.switch(peer_role)

    @property
    def active_peer(self) -> PeerConfig:
        return self.peers.active_peer

    # ═══════════════════════ Prompt 生成 ═══════════════════════

    def get_user_prompt_fragment(self) -> str:
        """生成用户画像的 prompt 注入片段。"""
        p = self.profile

        if p.total_interactions == 0:
            return ""

        lines = ["<user_profile>"]
        active_techs = [
            name for name, tp in sorted(
                p.technologies.items(),
                key=lambda x: x[1].usage_count, reverse=True
            )[:5]
            if tp.usage_count >= 2
        ]
        if active_techs:
            lines.append(f"  常用技术栈: {', '.join(active_techs)}")

        top_tools = sorted(
            p.tool_preferences.items(), key=lambda x: x[1], reverse=True
        )[:3]
        if top_tools:
            tools_str = ", ".join(f"{t}({int(c)}次)" for t, c in top_tools)
            lines.append(f"  偏好工具: {tools_str}")

        if p.common_workflows:
            lines.append(f"  常见工作流: {', '.join(p.common_workflows)}")

        if p.inferred_goals:
            lines.append(f"  当前目标: {', '.join(p.inferred_goals)}")

        lines.append("</user_profile>")
        return "\n".join(lines)

    def get_full_prompt_fragment(self) -> str:
        """生成完整的 prompt 注入片段（用户画像 + Peer 角色）。"""
        parts = [self.peers.get_peer_prompt_fragment()]
        user_fragment = self.get_user_prompt_fragment()
        if user_fragment:
            parts.append(user_fragment)
        return "\n".join(parts)

    # ═══════════════════════ 持久化 ═══════════════════════

    def save(self) -> None:
        """持久化用户画像到 JSON。"""
        data = {
            "preferred_language": self.profile.preferred_language,
            "communication_style": self.profile.communication_style,
            "active_hours": self.profile.active_hours,
            "technologies": {
                name: {
                    "name": tp.name,
                    "level": tp.level.value,
                    "usage_count": tp.usage_count,
                    "first_seen": tp.first_seen,
                    "last_seen": tp.last_seen,
                    "success_rate": tp.success_rate,
                    "common_errors": tp.common_errors,
                }
                for name, tp in self.profile.technologies.items()
            },
            "common_workflows": self.profile.common_workflows,
            "frequent_commands": self.profile.frequent_commands,
            "tool_preferences": self.profile.tool_preferences,
            "inferred_goals": self.profile.inferred_goals,
            "current_project": self.profile.current_project,
            "first_interaction": self.profile.first_interaction,
            "last_interaction": self.profile.last_interaction,
            "total_interactions": self.profile.total_interactions,
            "version": self.profile.version,
        }
        self._profile_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        """从 JSON 恢复用户画像。"""
        if not self._profile_path.exists():
            return
        try:
            data = json.loads(self._profile_path.read_text(encoding="utf-8"))
            self.profile.preferred_language = data.get("preferred_language", "")
            self.profile.communication_style = data.get("communication_style", "")
            self.profile.active_hours = data.get("active_hours", [])
            self.profile.common_workflows = data.get("common_workflows", [])
            self.profile.frequent_commands = data.get("frequent_commands", [])
            self.profile.tool_preferences = data.get("tool_preferences", {})
            self.profile.inferred_goals = data.get("inferred_goals", [])
            self.profile.current_project = data.get("current_project", "")
            self.profile.first_interaction = data.get("first_interaction", 0.0)
            self.profile.last_interaction = data.get("last_interaction", 0.0)
            self.profile.total_interactions = data.get("total_interactions", 0)
            self.profile.version = data.get("version", 1)

            for name, tdata in data.get("technologies", {}).items():
                self.profile.technologies[name] = TechnologyProfile(
                    name=tdata.get("name", name),
                    level=SkillLevel(tdata.get("level", "unknown")),
                    usage_count=tdata.get("usage_count", 0),
                    first_seen=tdata.get("first_seen", 0),
                    last_seen=tdata.get("last_seen", 0),
                    success_rate=tdata.get("success_rate", 1.0),
                    common_errors=tdata.get("common_errors", []),
                )
        except Exception:
            pass
