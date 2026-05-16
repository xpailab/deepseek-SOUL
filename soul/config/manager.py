"""配置管理器 — 统一加载、保存、热更新。

支持：
- 从 YAML/JSON 文件加载
- 环境变量覆盖（SOUL_ 前缀）
- 运行时热更新（部分字段）
- 配置版本迁移
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from soul.types import SOULConfig


class ConfigManager:
    """全局配置管理器。

    加载优先级：默认值 → 配置文件 → 环境变量 SOUL_* → 运行时覆盖
    """

    _instance: "ConfigManager | None" = None

    def __init__(self, config_path: str | None = None):
        self._config_path = Path(config_path) if config_path else self._default_path()
        self._config = SOULConfig()
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _default_path() -> Path:
        return Path.home() / ".soul" / "config.yaml"

    @property
    def config(self) -> SOULConfig:
        if not self._loaded:
            self.load()
        return self._config

    def load(self, path: str | None = None) -> SOULConfig:
        """加载配置。"""
        if path:
            self._config_path = Path(path)

        load_dotenv(Path.home() / ".soul" / ".env")

        if self._config_path.exists():
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._config = SOULConfig(**data)

        self._apply_env_overrides()
        self._loaded = True
        return self._config

    def save(self, path: str | None = None) -> None:
        """保存配置到文件。"""
        target = Path(path) if path else self._config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.dump(
                self._config.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def update(self, **kwargs: Any) -> None:
        """运行时更新配置（部分字段）。"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def get(self, path: str, default: Any = None) -> Any:
        """点号分隔路径获取配置值。例如 'llm.model'"""
        keys = path.split(".")
        value: Any = self._config
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖。"""
        overrides = {
            "SOUL_LLM_PROVIDER": "llm.provider",
            "SOUL_LLM_MODEL": "llm.model",
            "SOUL_LLM_API_KEY": "llm.api_key",
            "SOUL_LLM_API_BASE": "llm.api_base",
            "SOUL_LLM_MAX_TOKENS": "llm.max_tokens",
            "SOUL_GATEWAY_PORT": "gateway.port",
            "SOUL_GATEWAY_HOST": "gateway.host",
            "SOUL_MAX_CONCURRENT": "lane.max_concurrent",
            "SOUL_WORKSPACE_DIR": "memory.workspace_dir",
            "SOUL_DEBUG": "debug",
        }
        for env_var, config_path in overrides.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_by_path(config_path, value)

    def _set_by_path(self, path: str, value: str) -> None:
        """通过路径设置配置值。"""
        keys = path.split(".")
        obj = self._config
        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                return
        last_key = keys[-1]
        if hasattr(obj, last_key):
            field_type = type(getattr(obj, last_key))
            if field_type is bool:
                value = value.lower() in ("true", "1", "yes")
            elif field_type is int:
                value = int(value)
            elif field_type is float:
                value = float(value)
            setattr(obj, last_key, value)

    def reset(self) -> None:
        """重置为默认配置。"""
        self._config = SOULConfig()
        self._loaded = False
