"""DM 配对安全 — 防止未授权的私信访问。

默认 dm_policy="pairing"，未知发送者需要配对码才能使用。
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path


class PairingManager:
    """DM 配对管理器。

    安全模型:
    - 未知发送者首次 DM 时需提供配对码
    - 配对码 5 分钟有效
    - 配对后可永久或按时间限制访问
    - 支持白名单/黑名单管理
    """

    def __init__(self, data_dir: str = "~/.soul"):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._pending_codes: dict[str, tuple[str, float]] = {}  # code -> (user_id, expiry)
        self._paired_users: dict[str, dict[str, str]] = {}  # channel:user_id -> info
        self._whitelist: set[str] = set()
        self._blacklist: set[str] = set()
        self._code_timeout = 300  # 5 分钟

    def generate_code(self, user_id: str) -> str:
        """生成配对码。"""
        code = secrets.token_hex(3).upper()  # 6 字符
        self._pending_codes[code] = (user_id, time.time() + self._code_timeout)
        return code

    def approve(self, channel: str, code: str) -> bool:
        """批准配对码。"""
        entry = self._pending_codes.get(code.upper())
        if not entry:
            return False

        user_id, expiry = entry
        if time.time() > expiry:
            del self._pending_codes[code]
            return False

        key = f"{channel}:{user_id}"
        self._paired_users[key] = {
            "user_id": user_id,
            "channel": channel,
            "paired_at": time.time(),
        }
        del self._pending_codes[code]
        return True

    def check(self, channel: str, user_id: str) -> bool:
        """检查用户是否已配对。"""
        key = f"{channel}:{user_id}"

        if key in self._blacklist:
            return False

        if key in self._whitelist:
            return True

        return key in self._paired_users

    def revoke(self, channel: str, user_id: str) -> None:
        """撤销配对。"""
        key = f"{channel}:{user_id}"
        self._paired_users.pop(key, None)

    def add_whitelist(self, channel: str, user_id: str) -> None:
        self._whitelist.add(f"{channel}:{user_id}")

    def add_blacklist(self, channel: str, user_id: str) -> None:
        self._blacklist.add(f"{channel}:{user_id}")

    def list_paired(self) -> list[dict[str, str]]:
        return [
            {"key": k, **v}
            for k, v in self._paired_users.items()
        ]

    def cleanup_expired_codes(self) -> int:
        """清理过期的配对码。"""
        now = time.time()
        expired = [c for c, (_, e) in self._pending_codes.items() if now > e]
        for c in expired:
            del self._pending_codes[c]
        return len(expired)
