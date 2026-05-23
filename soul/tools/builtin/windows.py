"""Windows GUI 操作工具 — 窗口激活、模拟输入、发送按键。"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from soul.tools.registry import ToolDef
from soul.types import ToolRisk

# 查找 powershell.exe — 优先用 SystemRoot 环境变量
_PS_EXE = (
    shutil.which("powershell.exe")
    or shutil.which("powershell")
    or os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
)


class WindowsTool:
    """Windows GUI 自动化工具。"""

    NAME = "win"
    DESCRIPTION = (
        "Windows GUI 操作: 激活窗口、模拟键盘输入、发送文本。"
        "window_title 参数支持中文名，会自动尝试英文进程名。如'豆包'会同时搜索'Doubao'。"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["activate", "sendkeys", "type", "find_window", "screenshot", "click"],
                "description": "操作: activate/sendkeys/type/find_window/screenshot/click(点击坐标或按钮文本)",
            },
            "window_title": {
                "type": "string",
                "description": "窗口标题关键词（如 '豆包'、'Chrome'）",
            },
            "text": {
                "type": "string",
                "description": "要输入的文本",
            },
            "keys": {
                "type": "string",
                "description": "要发送的按键（如 '{ENTER}'、'^v'=Ctrl+V）",
            },
            "timeout": {
                "type": "integer",
                "description": "超时秒数",
                "default": 10,
            },
            "x": {
                "type": "integer",
                "description": "点击的屏幕 X 坐标（click 操作）",
            },
            "y": {
                "type": "integer",
                "description": "点击的屏幕 Y 坐标（click 操作）",
            },
            "button_text": {
                "type": "string",
                "description": "要点击的按钮/链接文本（click 操作，用 UIAutomation 定位）",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        pass

    async def execute(
        self,
        action: str,
        window_title: str = "",
        text: str = "",
        keys: str = "",
        timeout: int = 10,
        x: int = 0,
        y: int = 0,
        button_text: str = "",
    ) -> dict[str, Any]:
        """执行 Windows GUI 操作。"""
        import platform
        if platform.system() != "Windows":
            return {"error": "Windows GUI 工具仅在 Windows 系统可用", "success": False}
        try:
            if action == "activate":
                return await self._activate(window_title)
            elif action == "sendkeys":
                return await self._send_keys(window_title, keys)
            elif action == "type":
                return await self._type_text(window_title, text)
            elif action == "find_window":
                return await self._find_window(window_title)
            elif action == "screenshot":
                return await self._screenshot(window_title)
            elif action == "click":
                return await self._click(window_title, x, y, button_text)
            else:
                return {"error": f"未知操作: {action}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _run_ps(self, script: str, timeout: int = 10) -> dict[str, Any]:
        """运行 PowerShell 脚本。"""
        ps_path = tempfile.mktemp(suffix=".ps1")
        Path(ps_path).write_text(script, encoding="utf-8")

        try:
            # 用完整路径调 powershell.exe
            proc = await asyncio.create_subprocess_exec(
                _PS_EXE, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", ps_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            # 尝试 UTF-8，失败用 GBK
            for enc in ["utf-8", "gbk", "latin-1"]:
                try:
                    out = stdout.decode(enc)
                    if "\ufffd" not in out or len(out) < 100:
                        break
                except Exception:
                    continue
            err = stderr.decode("gbk", errors="replace")

            return {
                "exit_code": proc.returncode,
                "stdout": out[:5000],
                "stderr": err[:1000],
                "success": proc.returncode == 0,
            }
        finally:
            try:
                os.unlink(ps_path)
            except Exception:
                pass

    async def _activate(self, window_title: str) -> dict[str, Any]:
        """激活指定窗口，先按标题再按进程名查找。"""
        script = f'''
Add-Type @"
using System; using System.Runtime.InteropServices;
public class WinApi {{
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
    [DllImport("user32.dll")] public static extern bool EnumWindows(IntPtr lpEnumFunc, IntPtr lParam);
}}
"@
$title = "{window_title}"
# 先按窗口标题找
$procs = Get-Process | Where-Object {{ $_.MainWindowTitle -like "*$title*" -and $_.MainWindowHandle -ne 0 }}
if (-not $procs) {{
    # 没找到 → 按进程名找
    $procs = Get-Process -Name "*$title*" -ErrorAction SilentlyContinue | Where-Object {{ $_.MainWindowHandle -ne 0 }}
}}
if (-not $procs) {{ Write-Host "NOT_FOUND"; exit 1 }}
foreach ($p in $procs) {{
    $h = $p.MainWindowHandle
    if ($h -ne [IntPtr]::Zero) {{
        [WinApi]::ShowWindow($h, 9)
        Start-Sleep -Milliseconds 200
        [WinApi]::SetForegroundWindow($h)
        Start-Sleep -Milliseconds 200
        Write-Host "ACTIVATED:$($p.Id):$($p.MainWindowTitle)"
        exit 0
    }}
}}
Write-Host "NO_VALID_HANDLE"
exit 1
'''
        return await self._run_ps(script)

    async def _send_keys(self, window_title: str, keys: str) -> dict[str, Any]:
        """激活窗口并发送按键。"""
        script = f'''
Add-Type -AssemblyName System.Windows.Forms
$title = "{window_title}"
$keys = "{keys}"
$procs = Get-Process | Where-Object {{ $_.MainWindowTitle -like "*$title*" -and $_.MainWindowHandle -ne 0 }}
if (-not $procs) {{ $procs = Get-Process -Name "*$title*" -ErrorAction SilentlyContinue | Where-Object {{ $_.MainWindowHandle -ne 0 }} }}
if (-not $procs) {{ Write-Host "NOT_FOUND"; exit 1 }}
$h = $procs[0].MainWindowHandle
if ($h -eq [IntPtr]::Zero) {{ Write-Host "NO_HANDLE"; exit 1 }}
Add-Type @"
using System; using System.Runtime.InteropServices;
public class WinApi {{
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
}}
"@
[WinApi]::ShowWindow($h, 9)
[WinApi]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 300
[System.Windows.Forms.SendKeys]::SendWait($keys)
Write-Host "SENT:$keys"
exit 0
'''
        return await self._run_ps(script)

    async def _type_text(self, window_title: str, text: str) -> dict[str, Any]:
        """激活窗口并输入文本。"""
        script = f'''
Add-Type -AssemblyName System.Windows.Forms
$title = "{window_title}"
$procs = Get-Process | Where-Object {{ $_.MainWindowTitle -like "*$title*" -and $_.MainWindowHandle -ne 0 }}
if (-not $procs) {{ $procs = Get-Process -Name "*$title*" -ErrorAction SilentlyContinue | Where-Object {{ $_.MainWindowHandle -ne 0 }} }}
if (-not $procs) {{ Write-Host "NOT_FOUND"; exit 1 }}
$h = $procs[0].MainWindowHandle
if ($h -eq [IntPtr]::Zero) {{ Write-Host "NO_HANDLE"; exit 1 }}
Add-Type @"
using System; using System.Runtime.InteropServices;
public class WinApi {{
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
}}
"@
[WinApi]::ShowWindow($h, 9)
[WinApi]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 500
[System.Windows.Forms.SendKeys]::SendWait("{text}")
Write-Host "TYPED"
exit 0
'''
        return await self._run_ps(script)

    async def _find_window(self, title: str) -> dict[str, Any]:
        """查找窗口，先按标题再按进程名，最后列出建议。"""
        script = f'''
$title = "{title}"
# 先查带窗口标题的
$procs = Get-Process | Where-Object {{ $_.MainWindowTitle -like "*$title*" -and $_.MainWindowHandle -ne 0 }}
if ($procs) {{
    foreach ($p in $procs) {{ Write-Host "FOUND:PID=$($p.Id) TITLE=$($p.MainWindowTitle)" }}
    exit 0
}}
# 查进程名
$procs = Get-Process -Name "*$title*" -ErrorAction SilentlyContinue
if ($procs) {{
    Write-Host "PROCS:"
    foreach ($p in $procs) {{
        Write-Host "PID=$($p.Id) HWND=$($p.MainWindowHandle) NAME=$($p.Name) TITLE=$($p.MainWindowTitle)"
    }}
    exit 0
}}
# 都找不到 → 列出所有有窗口的进程作为参考
Write-Host "NOT_FOUND: 未找到 '$title'。当前有窗口的进程:"
Get-Process | Where-Object {{ $_.MainWindowHandle -ne 0 }} | Select-Object -First 10 | ForEach-Object {{
    Write-Host "  PID=$($_.Id) NAME=$($_.Name) TITLE=$($_.MainWindowTitle.Substring(0,[Math]::Min(40,$_.MainWindowTitle.Length)))"
}}
exit 1
'''
        return await self._run_ps(script)

    async def _screenshot(self, title: str = "") -> dict[str, Any]:
        """截取窗口截图（保存到工作空间）。"""
        script = f'''
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$screen = [System.Windows.Forms.Screen]::PrimaryScreen
$bmp = New-Object System.Drawing.Bitmap($screen.Bounds.Width, $screen.Bounds.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen(0, 0, 0, 0, $bmp.Size)
$g.Dispose()
$path = Join-Path $env:SOUL_WORKSPACE "screenshot_{title}.png"
$bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Host "SAVED:$path"
'''
        return await self._run_ps(script)

    async def _click(self, window_title: str, x: int, y: int, button_text: str) -> dict[str, Any]:
        """点击屏幕坐标或 UIAutomation 元素。"""
        if button_text:
            # UIAutomation 模式：查找并点击按钮
            script = f'''
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms, System.Drawing
$auto = [System.Windows.Automation.AutomationElement]
$target = "{button_text}"
$root = $auto::RootElement
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::NameProperty, $target)
try {{
    $el = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
    if ($el) {{
        $pt = $el.GetClickablePoint()
        [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point([int]$pt.X, [int]$pt.Y)
        Add-Type @"
using System; using System.Runtime.InteropServices;
public class Mouse {{
    [DllImport("user32.dll")] public static extern void mouse_event(int dwFlags, int dx, int dy, int dwData, int dwExtraInfo);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
}}
"@
        [Mouse]::SetCursorPos([int]$pt.X, [int]$pt.Y)
        Start-Sleep -Milliseconds 100
        [Mouse]::mouse_event(0x0002, 0, 0, 0, 0)
        Start-Sleep -Milliseconds 50
        [Mouse]::mouse_event(0x0004, 0, 0, 0, 0)
        Write-Host "CLICKED_TEXT:$target"
        exit 0
    }} else {{
        Write-Host "TEXT_NOT_FOUND:$target"
        exit 1
    }}
}} catch {{
    Write-Host "UA_ERROR:$($_.Exception.Message)"
    exit 1
}}
'''
        elif x > 0 and y > 0:
            # 坐标模式
            script = f'''
Add-Type @"
using System; using System.Runtime.InteropServices;
public class Mouse {{
    [DllImport("user32.dll")] public static extern void mouse_event(int dwFlags, int dx, int dy, int dwData, int dwExtraInfo);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
}}
"@
[Mouse]::SetCursorPos({x}, {y})
Start-Sleep -Milliseconds 100
[Mouse]::mouse_event(0x0002, 0, 0, 0, 0)
Start-Sleep -Milliseconds 50
[Mouse]::mouse_event(0x0004, 0, 0, 0, 0)
Write-Host "CLICKED:{x},{y}"
exit 0
'''
        else:
            # 无目标 → 先查找窗口再点击其中心
            script = f'''
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$title = "{window_title}"
if ($title) {{
    $procs = Get-Process | Where-Object {{ $_.MainWindowTitle -like "*$title*" -and $_.MainWindowHandle -ne 0 }}
    if (-not $procs) {{ $procs = Get-Process -Name "*$title*" -ErrorAction SilentlyContinue | Where-Object {{ $_.MainWindowHandle -ne 0 }} }}
    if ($procs) {{
        $h = $procs[0].MainWindowHandle
        Add-Type @"
using System; using System.Runtime.InteropServices;
public class Win {{
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
    [DllImport("user32.dll")] public static extern void mouse_event(int dwFlags, int dx, int dy, int dwData, int dwExtraInfo);
}}
public struct RECT {{ public int L,T,R,B; }}
"@
        [Win]::SetForegroundWindow($h)
        Start-Sleep -Milliseconds 300
        $r = New-Object RECT
        [Win]::GetWindowRect($h, [ref]$r)
        $cx = ($r.L + $r.R) / 2
        $cy = ($r.T + $r.B) / 2
        [Win]::SetCursorPos([int]$cx, [int]$cy)
        Start-Sleep -Milliseconds 100
        [Win]::mouse_event(0x0002, 0, 0, 0, 0)
        Start-Sleep -Milliseconds 50
        [Win]::mouse_event(0x0004, 0, 0, 0, 0)
        Write-Host "CLICKED_CENTER:$($procs[0].MainWindowTitle)"
        exit 0
    }}
}}
Write-Host "NO_TARGET"
exit 1
'''
        return await self._run_ps(script)

    @classmethod
    def to_tool_def(cls) -> ToolDef:
        return ToolDef(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            handler=cls().execute,
            parameters=cls.PARAMETERS,
            risk=ToolRisk.MEDIUM,
            requires_approval=False,
            timeout_seconds=15,
            tags=["windows", "gui", "automation"],
        )
