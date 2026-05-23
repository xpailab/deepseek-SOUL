"""文件操作工具 — 在工作空间内安全读写文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soul.tools.registry import ToolDef
from soul.types import ToolRisk


class FileTool:
    """文件读写工具。"""

    NAME = "file"
    DESCRIPTION = "在工作空间中安全地读取和写入文件。"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "edit", "delete", "list", "exists", "mkdir"],
                "description": "操作类型: read, write, edit, delete, list, exists, mkdir",
            },
            "file_path": {
                "type": "string",
                "description": "文件路径（相对于工作空间或绝对路径）",
            },
            "content": {
                "type": "string",
                "description": "写入内容（write/edit 操作需要）",
            },
            "old_string": {
                "type": "string",
                "description": "要替换的旧文本（edit 操作需要）",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的新文本（edit 操作需要）",
            },
            "offset": {
                "type": "integer",
                "description": "读取起始行（read 操作可选）",
            },
            "limit": {
                "type": "integer",
                "description": "读取行数上限（read 操作可选）",
            },
        },
        "required": ["action", "file_path"],
    }

    def __init__(self, workspace_dir: str = "~/.soul/workspace"):
        self.workspace = Path(workspace_dir).expanduser().resolve()

    async def execute(
        self,
        action: str,
        file_path: str,
        content: str = "",
        old_string: str = "",
        new_string: str = "",
        offset: int = 0,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """执行文件操作。"""
        path = self._resolve_path(file_path)

        try:
            if action == "read":
                return await self._read(path, offset, limit)
            elif action == "write":
                return await self._write(path, content)
            elif action == "edit":
                return await self._edit(path, old_string, new_string)
            elif action == "delete":
                return await self._delete(path)
            elif action == "list":
                return await self._list(path)
            elif action == "exists":
                return {"exists": path.exists(), "path": str(path)}
            elif action == "mkdir":
                return await self._mkdir(path)
            else:
                return {"error": f"未知操作: {action}", "success": False}
        except PermissionError as e:
            return {"error": f"权限不足: {e}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _read(self, path: Path, offset: int, limit: int) -> dict[str, Any]:
        if not path.exists():
            return {"error": f"文件不存在: {path}", "success": False}
        if path.is_dir():
            return {"error": f"路径是目录: {path}", "success": False}

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        selected = lines[offset : offset + limit]
        return {
            "content": "".join(selected),
            "total_lines": total_lines,
            "offset": offset,
            "lines_returned": len(selected),
            "path": str(path),
            "success": True,
        }

    async def _write(self, path: Path, content: str) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "bytes_written": len(content.encode("utf-8")),
            "path": str(path),
            "success": True,
        }

    async def _edit(self, path: Path, old: str, new: str) -> dict[str, Any]:
        if not path.exists():
            return {"error": f"文件不存在: {path}", "success": False}

        content = path.read_text(encoding="utf-8")
        if old not in content:
            return {"error": "old_string 未在文件中找到", "success": False}

        count = content.count(old)
        new_content = content.replace(old, new, 1) if count > 1 else content.replace(old, new)
        path.write_text(new_content, encoding="utf-8")
        return {
            "replacements": 1 if count > 1 else count,
            "total_occurrences": count,
            "path": str(path),
            "success": True,
        }

    async def _delete(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"error": f"文件不存在: {path}", "success": False}
        path.unlink()
        return {"deleted": str(path), "success": True}

    async def _list(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"error": f"目录不存在: {path}", "success": False}
        if not path.is_dir():
            return {"error": f"不是目录: {path}", "success": False}

        items = []
        for item in sorted(path.iterdir()):
            items.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return {"items": items, "count": len(items), "path": str(path), "success": True}

    async def _mkdir(self, path: Path) -> dict[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        return {"created": str(path), "success": True}

    def _resolve_path(self, file_path: str) -> Path:
        """解析路径。允许写入任意非系统路径，不限于工作空间。"""
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        resolved = path.resolve()

        # 阻止写入系统关键目录
        system_paths = [
            Path("/etc"), Path("/boot"), Path("/sys"), Path("/proc"), Path("/dev"),
            Path("/bin"), Path("/sbin"), Path("/usr/bin"), Path("/usr/sbin"),
            Path("C:/Windows"), Path("C:/Program Files"), Path("C:/ProgramData"),
        ]
        for sys_path in system_paths:
            try:
                resolved.relative_to(sys_path)
                raise PermissionError(f"不能写入系统目录: {resolved}")
            except ValueError:
                pass

        return resolved

    @classmethod
    def to_tool_def(cls) -> ToolDef:
        return ToolDef(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            handler=cls().execute,
            parameters=cls.PARAMETERS,
            risk=ToolRisk.MEDIUM,
            requires_approval=False,
            timeout_seconds=30,
            max_retries=1,
            tags=["file", "io", "read", "write"],
        )
