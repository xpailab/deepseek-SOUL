"""检查点系统 — 长任务断点续跑。

长任务（20+ 步骤，15+ 分钟）中途崩溃时，所有进度丢失。此模块在每步完成后
自动持久化执行状态，下次启动时可从断点继续。

存储: ~/.soul/checkpoints/<session_id>.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    """执行状态快照。"""

    session_id: str
    task: str
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    current_step_index: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    ruled_out: list[str] = field(default_factory=list)
    tool_results_summary: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.plan_steps if s.get("completed", False))

    @property
    def total_steps(self) -> int:
        return len(self.plan_steps)

    @property
    def is_complete(self) -> bool:
        return self.completed_steps >= self.total_steps > 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "plan_steps": self.plan_steps,
            "current_step_index": self.current_step_index,
            "attempts": self.attempts[-20:],  # 只保留最后 20 条尝试
            "findings": self.findings[-10:],
            "ruled_out": self.ruled_out[-10:],
            "tool_results_summary": self.tool_results_summary[-20:],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Checkpoint:
        return cls(
            session_id=d.get("session_id", ""),
            task=d.get("task", ""),
            plan_steps=d.get("plan_steps", []),
            current_step_index=d.get("current_step_index", 0),
            attempts=d.get("attempts", []),
            findings=d.get("findings", []),
            ruled_out=d.get("ruled_out", []),
            tool_results_summary=d.get("tool_results_summary", []),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            version=d.get("version", 1),
        )


class CheckpointManager:
    """检查点管理器。

    使用方式:
        cpm = CheckpointManager()
        cpm.save(session_id, plan, working_memory)
        # ... 崩溃 ...
        cp = cpm.load_latest()
        if cp: agent.resume_from(cp)
    """

    def __init__(self, checkpoints_dir: str = "~/.soul/checkpoints"):
        self.dir = Path(checkpoints_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    # ═══ 保存 ═══

    def save(
        self,
        session_id: str,
        task: str = "",
        plan_steps: list[dict] | None = None,
        working_memory: Any = None,
        tool_summaries: list[str] | None = None,
    ) -> str:
        """保存检查点。"""
        # 从工作记忆提取状态
        attempts = []
        findings = []
        ruled_out = []
        if working_memory:
            attempts = working_memory.attempts[-20:]
            findings = working_memory.findings[-10:]
            ruled_out = working_memory.ruled_out[-10:]

        cp = Checkpoint(
            session_id=session_id,
            task=task,
            plan_steps=plan_steps or [],
            current_step_index=sum(1 for s in (plan_steps or []) if s.get("completed", False)),
            attempts=attempts,
            findings=findings,
            ruled_out=ruled_out,
            tool_results_summary=tool_summaries or [],
        )

        filepath = self.dir / f"{session_id}.json"
        data = cp.to_dict()
        # 原子写入
        tmp_path = filepath.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(filepath)
        return str(filepath)

    # ═══ 加载 ═══

    def load(self, session_id: str) -> Checkpoint | None:
        """加载指定会话的检查点。"""
        filepath = self.dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def load_latest(self) -> Checkpoint | None:
        """加载最近的不完整检查点。"""
        incomplete = self.list_incomplete()
        if not incomplete:
            return None
        return incomplete[0]

    def list_incomplete(self) -> list[Checkpoint]:
        """列出所有未完成的检查点（按时间倒序）。"""
        results = []
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                cp = Checkpoint.from_dict(data)
                if not cp.is_complete:
                    results.append(cp)
            except (json.JSONDecodeError, KeyError):
                pass
        return results

    # ═══ 管理 ═══

    def mark_complete(self, session_id: str) -> None:
        """标记检查点为完成——删除检查点文件。"""
        filepath = self.dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()

    def clean_old(self, max_age_days: int = 7) -> int:
        """清理过期检查点。"""
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for f in self.dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        return removed

    def get_resume_context(self, cp: Checkpoint) -> str:
        """生成断点续跑提示——注入到 system prompt。"""
        lines = [
            "\n## 断点续跑 — 从上次中断处继续",
            f"原始任务: {cp.task[:200]}",
            f"已完成: {cp.completed_steps}/{cp.total_steps} 步骤",
        ]
        if cp.plan_steps:
            lines.append("\n### 已完成的步骤:")
            for s in cp.plan_steps:
                if s.get("completed"):
                    status = "✓" if s.get("success") else "✗"
                    lines.append(f"  {status} {s.get('action', '')} → {s.get('result_summary', '')[:100]}")
            # 下一步
            remaining = [s for s in cp.plan_steps if not s.get("completed")]
            if remaining:
                lines.append(f"\n### 下一步: {remaining[0].get('action', '继续执行')}")

        if cp.findings:
            lines.append("\n### 之前发现:")
            for f in cp.findings[:5]:
                lines.append(f"  - {f}")

        if cp.ruled_out:
            lines.append("\n### 之前排除的方向:")
            for r in cp.ruled_out[:5]:
                lines.append(f"  - ✗ {r}")

        lines.append("\n请继续完成剩余步骤，不要重复已完成的步骤。")
        return "\n".join(lines)
