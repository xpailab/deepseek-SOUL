# DeepSoul — 下一代 AI Agent 框架

融合 **OpenClaw** 的编排能力与 **Hermes Agent** 的自我进化机制，
提供更简洁、更强大的 AI Agent 开发体验。

## 核心特性

### 超越两者的设计
| 特性 | OpenClaw | Hermes Agent | DeepSoul |
|------|----------|-------------|----------|
| 并发调度 | 双层 Lane Queue (6 模式) | 基本并发 | **Lane Queue 2.0 (7 模式 + 自适应)** |
| 记忆系统 | 单层文件注入 | 三层 (冻结+技能+FTS5) | **四层 (+预测记忆)** |
| 技能系统 | 人工编写 | 自动生成 + GEPA 进化 | **自动生成 + GEPA 2.0 + 语义匹配** |
| 网关 | 20+ 通道 | 6+ 通道 | **统一协议 + WebSocket** |
| 安全 | 沙箱 + DM 配对 | 零遥测 + 命令审批 | **四层防护 + 审计** |
| MLOps | 无 | 批量轨迹 + RL 环境 | **完整管道 (生成→压缩→评估)** |
| 部署 | npm 安装 | curl 一键 | **一键安装 + Web UI** |

### 架构亮点

```
用户消息 → Session Lane (串行) → Global Lane (并发4)
         → Agent 核心循环
         → LLM 推理 (DeepSeek/Claude/OpenAI)
         → 工具调用 (安全检查 → 执行 → 重试)
         → 记忆更新 (4层)
         → 技能进化 (GEPA)
         → 回复用户
```

## 快速开始

### 安装
```bash
# 一条命令安装
curl -fsSL https://raw.githubusercontent.com/deepseek-SOUL/main/scripts/install.sh | bash

# 或从源码安装
git clone https://github.com/deepseek-SOUL.git
cd deepseek-SOUL
pip install -e ".[all]"
```

### 配置
```bash
# 交互式配置
soul config

# 或编辑配置文件
vim ~/.soul/config.yaml
```

### 使用
```bash
# 交互式对话
soul chat

# 单次执行
soul run "帮我创建一个 FastAPI 项目"

# 启动网关
soul gateway --port 18789

# 查看状态
soul status

# 诊断检查
soul doctor

# 启动 Web UI
python -m web.app
```

## 项目结构

```
deepseek-SOUL/
├── soul/                    # 核心引擎
│   ├── engine/              # Agent 执行循环 + Lane Queue + 会话
│   ├── memory/              # 4层记忆系统
│   ├── skills/              # 技能系统 + GEPA 进化
│   ├── tools/               # 工具系统 + 安全护栏
│   ├── prompt/              # Prompt 构建 + 缓存 + 压缩
│   ├── llm/                 # LLM 适配器 (DeepSeek/Claude/OpenAI)
│   ├── gateway/             # 统一消息网关
│   ├── safety/              # 沙箱 + 配对 + 审计
│   ├── cron/                # 定时任务调度
│   ├── mlops/               # MLOps 训练管道
│   ├── config/              # 配置管理
│   └── types.py             # 核心类型定义
├── skills/bundled/          # 内置技能
├── web/                     # Web UI (FastAPI)
├── scripts/                 # 安装脚本
├── docker/                  # Docker 配置
└── tests/                   # 测试
```

## 配置示例

```yaml
# ~/.soul/config.yaml
llm:
  provider: deepseek
  model: deepseek-v4-pro
  api_key: "sk-..."

lane:
  max_concurrent: 4
  default_mode: adaptive

memory:
  workspace_dir: "~/.soul/workspace"
  predictive_enabled: true

skill:
  auto_generate: true
  gepa_enabled: true

gateway:
  port: 18789
  dm_policy: pairing
```

## 许可证

MIT License
