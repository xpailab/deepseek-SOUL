"""安全审计器 — 记录和审查所有敏感操作。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Auditor:
    """安全审计器。

    记录:
    - 所有工具调用及其参数
    - 文件访问记录
    - 命令执行记录
    - 安全事件（拦截、拒绝等）
    """

    def __init__(self, log_dir: str = "~/.soul/logs"):
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = 10

    def record(
        self,
        event_type: str,
        details: dict[str, Any],
        severity: str = "info",
    ) -> None:
        """记录审计事件。"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "severity": severity,
            "details": details,
        }
        self._buffer.append(event)

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        session_id: str,
    ) -> None:
        """记录工具调用。"""
        self.record(
            "tool_call",
            {
                "tool": tool_name,
                "arguments": {k: str(v)[:200] for k, v in arguments.items()},
                "result": result[:500],
                "session_id": session_id,
            },
        )

    def record_safety_block(
        self,
        tool_name: str,
        reason: str,
        arguments: dict[str, Any],
    ) -> None:
        """记录安全拦截。"""
        self.record(
            "safety_block",
            {
                "tool": tool_name,
                "reason": reason,
                "arguments": str(arguments)[:200],
            },
            severity="warning",
        )

    def record_file_access(
        self,
        filepath: str,
        action: str,
        session_id: str,
    ) -> None:
        """记录文件访问。"""
        self.record(
            "file_access",
            {
                "path": filepath,
                "action": action,
                "session_id": session_id,
            },
        )

    def flush(self) -> None:
        """将缓冲写入磁盘。"""
        if not self._buffer:
            return

        date_str = time.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            for event in self._buffer:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        self._buffer.clear()

    def query(
        self,
        event_type: str = "",
        severity: str = "",
        limit: int = 50,
        date: str = "",
    ) -> list[dict[str, Any]]:
        """查询审计日志。"""
        if not date:
            date = time.strftime("%Y-%m-%d")

        log_file = self.log_dir / f"audit_{date}.jsonl"
        if not log_file.exists():
            return []

        results: list[dict[str, Any]] = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event_type and event.get("type") != event_type:
                        continue
                    if severity and event.get("severity") != severity:
                        continue
                    results.append(event)
                    if len(results) >= limit:
                        break
                except json.JSONDecodeError:
                    continue

        return results

    def get_security_report(self) -> dict[str, Any]:
        """生成安全报告。"""
        today = time.strftime("%Y-%m-%d")
        events = self.query(date=today)

        blocks = [e for e in events if e["type"] == "safety_block"]
        tool_calls = [e for e in events if e["type"] == "tool_call"]

        return {
            "date": today,
            "total_events": len(events),
            "safety_blocks": len(blocks),
            "tool_calls": len(tool_calls),
            "block_rate": round(len(blocks) / max(1, len(tool_calls)) * 100, 1),
            "recent_blocks": [
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(b["timestamp"])),
                    "tool": b["details"].get("tool", ""),
                    "reason": b["details"].get("reason", ""),
                }
                for b in blocks[-5:]
            ],
        }
