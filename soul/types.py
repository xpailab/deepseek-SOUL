"""核心类型定义 — 整个框架的 Pydantic 模型基础。

所有模块共享这些类型，确保类型安全和接口一致性。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ═══════════════════════════════════════════════════════════════
# ID 生成
# ═══════════════════════════════════════════════════════════════

def gen_id(prefix: str = "") -> str:
    """生成唯一 ID，可带前缀。"""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


def gen_hash(content: str, length: int = 8) -> str:
    """生成内容哈希。"""
    return hashlib.sha256(content.encode()).hexdigest()[:length]


# ═══════════════════════════════════════════════════════════════
# 枚举类型
# ═══════════════════════════════════════════════════════════════

class AgentState(str, Enum):
    """Agent 运行状态。"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    STREAMING = "streaming"
    COMPRESSING = "compressing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class QueueMode(str, Enum):
    """队列模式 — 从 OpenClaw 6 模式改进为 7 模式。

    新增 `adaptive` 模式：根据消息优先级和 Agent 状态自动选择最优处理方式。
    """
    STEER = "steer"              # 立即注入当前流式输出
    FOLLOWUP = "followup"         # 等当前回合结束后排队
    COLLECT = "collect"           # 合并积压消息为一条（默认）
    STEER_BACKLOG = "steer_backlog"  # 注入 + 保留副本后续处理
    INTERRUPT = "interrupt"       # 立即中断当前运行
    QUEUE = "queue"               # 标准 FIFO 排队
    ADAPTIVE = "adaptive"         # 智能选择最优模式 [SOUL 创新]


class SandboxMode(str, Enum):
    """沙箱模式。"""
    LOCAL = "local"           # 本地进程（默认）
    DOCKER = "docker"         # Docker 容器隔离
    SSH = "ssh"               # SSH 远程执行
    NONE = "none"             # 无沙箱（完整权限）


class SkillType(str, Enum):
    """技能类型。"""
    BUNDLED = "bundled"       # 内置捆绑
    MANAGED = "managed"       # 从注册中心安装
    WORKSPACE = "workspace"   # 工作空间自定义
    EVOLVED = "evolved"       # 自动进化生成 [SOUL 创新]


class MemoryLayer(str, Enum):
    """记忆层级。"""
    FROZEN = "frozen"         # Layer 1: 冻结快照
    PROCEDURAL = "procedural" # Layer 2: 程序技能
    INDEXED = "indexed"       # Layer 3: FTS5 + LLM 混合
    PREDICTIVE = "predictive" # Layer 4: 预测记忆 [SOUL 创新]


class ToolRisk(str, Enum):
    """工具风险等级。"""
    SAFE = "safe"             # 只读、无副作用
    LOW = "low"               # 轻微副作用
    MEDIUM = "medium"         # 可能修改文件
    HIGH = "high"             # 可能执行代码
    CRITICAL = "critical"     # 系统级操作


class MessageRole(str, Enum):
    """消息角色。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ═══════════════════════════════════════════════════════════════
# 消息模型
# ═══════════════════════════════════════════════════════════════

class ToolCall(BaseModel):
    """工具调用。"""
    id: str = Field(default_factory=lambda: gen_id("tc_"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """工具执行结果。"""
    call_id: str
    name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0
    classification: Literal["success", "partial", "denied", "failure", "timeout", "rate_limited"] = "success"


class Message(BaseModel):
    """统一消息模型。"""
    id: str = Field(default_factory=lambda: gen_id("msg_"))
    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    reasoning_content: str = ""  # DeepSeek 思考模式：内部推理过程，发回 API 时必须原样保留
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            return ""
        return v


class StreamChunk(BaseModel):
    """流式输出块。"""
    content: str = ""
    reasoning_content: str = ""  # DeepSeek 思考模式
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None  # 工具执行结果（前端放可折叠区）
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════
# 会话模型
# ═══════════════════════════════════════════════════════════════

class SessionState(BaseModel):
    """会话状态 — 可序列化、可恢复。"""
    session_id: str = Field(default_factory=lambda: gen_id("sess_"))
    session_key: str = "main"
    agent_state: AgentState = AgentState.IDLE
    messages: list[Message] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)
    message_count: int = 0
    token_count: int = 0
    sandbox_mode: SandboxMode = SandboxMode.LOCAL
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 配置模型
# ═══════════════════════════════════════════════════════════════

class LLMConfig(BaseModel):
    """LLM 提供商配置。"""
    provider: str = "deepseek"  # deepseek, claude, openai
    model: str = "deepseek-v4-pro"
    api_key: str = ""
    api_base: str = ""
    max_tokens: int = 8192
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    timeout: float = Field(default=300.0, gt=0)  # 增加到5分钟以支持复杂任务
    max_retries: int = Field(default=3, ge=0)


class LaneConfig(BaseModel):
    """Lane 队列配置。"""
    max_concurrent: int = 4          # Global lane 并发上限
    session_concurrent: int = 1      # Session lane 并发上限
    subagent_concurrent: int = 8     # Sub-agent lane 并发上限
    cron_concurrent: int = 2         # Cron lane 并发上限
    default_mode: QueueMode = QueueMode.ADAPTIVE
    debounce_ms: int = 1000          # 防抖延迟
    cap: int = Field(default=20, ge=1)   # 队列容量上限
    drop_policy: Literal["old", "new", "summarize"] = "summarize"


class MemoryConfig(BaseModel):
    """记忆系统配置。"""
    workspace_dir: str = "~/.soul/workspace"
    frozen_max_chars: int = 2200     # MEMORY.md 上限
    user_max_chars: int = 1375       # USER.md 上限
    fts_db_path: str = "~/.soul/memory.db"
    honcho_enabled: bool = True
    predictive_enabled: bool = True  # Layer 4 预测记忆
    auto_compress_threshold: float = 0.8  # 80% 容量时自动压缩


class SkillConfig(BaseModel):
    """技能系统配置。"""
    skills_dir: str = "~/.soul/skills"
    auto_generate: bool = True       # 自动从轨迹生成技能
    gepa_enabled: bool = True        # GEPA 进化引擎
    gepa_max_iterations: int = 10
    gepa_population_size: int = 8
    skill_max_size_kb: int = 15


class GatewayConfig(BaseModel):
    """网关配置。"""
    port: int = Field(default=18789, ge=1, le=65535)
    host: str = "0.0.0.0"
    channels: list[str] = Field(default_factory=lambda: ["cli"])
    dm_policy: Literal["open", "pairing", "whitelist"] = "pairing"
    websocket_enabled: bool = True


class SandboxConfig(BaseModel):
    """沙箱配置。"""
    default_mode: SandboxMode = SandboxMode.LOCAL
    docker_image: str = "soul-sandbox:latest"
    readonly_root: bool = True
    memory_limit: str = "512m"
    cpu_limit: str = "1.0"
    network_enabled: bool = False
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(
        default_factory=lambda: ["/etc/passwd", "/etc/shadow", "~/.ssh"]
    )


class MLOpsConfig(BaseModel):
    """MLOps 训练配置。"""
    output_dir: str = "~/.soul/training"
    max_trajectories: int = 1000
    parallel_workers: int = 4
    checkpoint_enabled: bool = True
    output_format: Literal["sharegpt", "openai", "claude"] = "sharegpt"


class SOULConfig(BaseModel):
    """DeepSoul 顶层配置 — 单一配置文件。

    所有子系统配置统一管理，一份配置文件覆盖全部功能。
    """
    version: str = "0.1.0"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    lane: LaneConfig = Field(default_factory=LaneConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skill: SkillConfig = Field(default_factory=SkillConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    debug: bool = False
    verbose: bool = False
    soul_file: str = "~/.soul/SOUL.md"  # Agent 人格定义
    identity_file: str = "~/.soul/IDENTITY.md"


# ═══════════════════════════════════════════════════════════════
# Skill 模型
# ═══════════════════════════════════════════════════════════════

class SkillMeta(BaseModel):
    """技能元数据。"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    type: SkillType = SkillType.BUNDLED
    triggers: list[str] = Field(default_factory=list)  # 触发条件
    dependencies: list[str] = Field(default_factory=list)
    gepa_generation: int = 0  # GEPA 进化代数
    fitness_score: float = 0.0  # 综合适应度分数
    usage_count: int = 0
    success_rate: float = 1.0


class Skill(BaseModel):
    """技能定义。"""
    meta: SkillMeta
    content: str  # SKILL.md 内容
    hash: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def model_post_init(self, __context: Any) -> None:
        if not self.hash:
            self.hash = gen_hash(self.content)


# ═══════════════════════════════════════════════════════════════
# 记忆条目模型
# ═══════════════════════════════════════════════════════════════

class MemoryEntry(BaseModel):
    """记忆条目。"""
    id: str = Field(default_factory=lambda: gen_id("mem_"))
    layer: MemoryLayer
    content: str
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5  # 0-1 重要性
    access_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 事件模型
# ═══════════════════════════════════════════════════════════════

class AgentEvent(BaseModel):
    """Agent 生命周期事件。"""
    event_type: str
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 轨迹模型 (MLOps)
# ═══════════════════════════════════════════════════════════════

class TrajectoryStep(BaseModel):
    """轨迹中的单步。"""
    step_index: int
    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    thinking: str | None = None
    duration_ms: float = 0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """完整执行轨迹。"""
    id: str = Field(default_factory=lambda: gen_id("traj_"))
    session_id: str
    task: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    success: bool = False
    total_duration_ms: float = 0
    total_tokens: int = 0
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
