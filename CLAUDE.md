## 项目背景

DeepSoul — 下一代 AI Agent 框架，融合 OpenClaw 编排能力 + Hermes 自我进化。

仓库: https://github.com/xpailab/deepseek-SOUL

### 技术栈
- Python 3.11+, Pydantic v2, asyncio, aiosqlite
- DeepSeek/Claude/OpenAI API 适配
- Typer CLI + FastAPI Web UI
- SQLite FTS5 全文搜索

### 核心模块
- soul/engine/ — Agent 循环 + Lane Queue 2.0 并发调度
- soul/memory/ — 4层记忆 (冻结→技能→FTS5→预测)
- soul/skills/ — 自动技能生成 + GEPA 进化引擎
- soul/tools/ — 工具注册 + 安全护栏 + 结果分类 + 重试
- soul/llm/ — 多LLM适配器 (DeepSeek/Claude/OpenAI)
- soul/prompt/ — Prompt构建 + 前缀缓存 + 上下文压缩
- soul/gateway/ — 统一消息网关 + WebSocket
- soul/safety/ — 沙箱隔离 + DM配对 + 审计
- soul/cron/ — 定时任务调度
- soul/mlops/ — 轨迹生成 + 压缩 + LLM评估

### 安装运行
```bash
pip install -e ".[all]"
soul chat      # 交互式对话
soul gateway   # 启动网关
```

### 注意事项
不能删除和修改除了该目录下的任何文件

### 沟通语言
使用中文和我沟通

## 代码检查
每修改一次代码，都需要进行至少一次检查，确保没有错误。