# DeepSoul — 下一代自我进化 AI Agent 框架

> **本项目 99% 以上由 [deepseek-v4-pro](https://platform.deepseek.com) 构建。**  
> 创建初衷是验证 DeepSeek v4 能否自主构建一个媲美 OpenClaw 的生产级 Agent 框架。结论是**完全没有问题**——v4 的编码能力确实一流，配合其定价策略，整个项目耗时约两周（主要是周末），**总成本不到 50 元人民币**。  
> [English →](README.md)

DeepSoul 是一个**越用越聪明**的 AI Agent —— 它能操控电脑、写代码、管项目、跑定时任务，并从每次任务中学习、自我改进。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 能做什么

### 日常开发
```bash
deepsoul run "创建一个带用户登录、JWT 认证和数据库模型的 FastAPI 项目"
deepsoul run "把 soul/engine/agent.py 重构为更小的方法"
deepsoul run "为这个项目写 Dockerfile 和 docker-compose.yml"
deepsoul run "找出这个代码库中所有的 SQL 注入漏洞"
```

### 定时自动化
用自然语言描述任务，自动转换为 cron 定时任务：
```bash
deepsoul run "每天早上9点检查服务器日志，把错误摘要发送到 Telegram"
deepsoul run "每小时把数据库备份到 /backup"
```

### 跨平台消息网关
一个网关连接所有聊天平台：
```bash
deepsoul gateway --port 18789
# 支持: QQ / 微信 / 钉钉 / 飞书 / Telegram / REST API / WebSocket
# 所有平台共享同一会话状态
```

### 自我进化——真正越用越聪明
每次任务完成后，Agent **主动学习**并应用到未来任务：
- **教训提取**: 自动提取每次任务的错误、发现、修复方案
- **防御规则**: 同一错误模式 ≥2 次 → 自动生成预警，下次任务前置提醒
- **结构化日报**: L1 记忆带时间戳、文件列表、发现、状态标记
- **主动教练**: 新会话自动注入相关历史教训——踩过的坑不再踩第二次
- **技能生成**: 同类任务成功 3 次以上 → 自动生成 SKILL.md → 下次复用
- **GEPA 进化**: 遗传算法持续优化 prompt → 准确率↑，成本↓

### 角色系统——10 种内置身份
每条消息自动检测任务类型，匹配最合适的角色：
```
你: "帮我写模块的pytest测试" → 🧪 测试工程师
你: "分析销售趋势"          → 📊 数据分析师
你: "给新手解释递归原理"    → 📚 老师/教育者
你: "部署到K8s集群"         → ⚙️ DevOps工程师
```
每个角色自带技能、上下文规则和专业领域知识。说"帮我创建一个xxx角色"即可对话式创建新角色。

### 训练你自己的 Agent
```bash
deepsoul train tasks.txt --count 1000 --workers 8
```

---

## 模型架构

| 组件 | 模型 | 占比 |
|------|------|------|
| 核心 Agent 循环、推理、工具使用、代码生成 | **deepseek-v4-pro** | 98% |
| 多模态理解（图片、文档） | **kimi-k2.5** | 2% |

DeepSeek v4 Pro 提供核心推理能力，成本仅为同类模型的 1/10。Kimi K2.5 在需要图片和文档理解时辅助。

---

## 5 分钟快速开始

### 1. 安装
```bash
curl -fsSL https://raw.githubusercontent.com/xpailab/deepseek-SOUL/main/scripts/install.sh | bash

# 或从源码安装
git clone https://github.com/xpailab/deepseek-SOUL.git
cd deepseek-SOUL
pip install -e ".[all]"
```

### 2. 配置 API Key
```bash
deepsoul config llm.provider deepseek
deepsoul config llm.api_key sk-your-key-here
deepsoul config llm.model deepseek-v4-pro
# 支持: deepseek / claude / openai
```

### 3. 开始使用
```bash
deepsoul chat                         # 交互式对话
deepsoul run "创建一个 React 项目"      # 单次任务
deepsoul status                        # 查看状态
deepsoul doctor                        # 系统诊断
```

### 4. 启动网关（可选）
```bash
deepsoul gateway --port 18789
# 浏览器打开 http://localhost:18789 进入 Web 界面
# API 文档: http://localhost:18789/docs
```

---

## 内置工具

| 工具 | 能力 | 风险等级 |
|------|------|---------|
| `bash` | Shell 命令、安装软件、管理进程 | 高（有护栏） |
| `file` | 文件读写/编辑/删除、创建目录 | 中（路径沙箱） |
| `web` | HTTP 请求、网页搜索、API 调用 | 低（禁内网 IP） |
| `browser` | 浏览器自动化、网页抓取 | 中 |
| `win` | Windows GUI 自动化 | 中 |

所有工具调用经过 4 层安全防护：参数校验 → 路径沙箱 → 命令审批 → 审计记录。

---

## 它是如何变聪明的

```
第一次:  deepsoul run "把 Django 部署到服务器"
  → 15 分钟, 8000 tokens → 自动分析执行过程...

第二次:  deepsoul run "把 Django 部署到服务器"
  → 匹配到学到的技能, 5 分钟, 3000 tokens

第十次:  deepsoul run "把 Django 部署到服务器"
  → GEPA 已优化 5 代, 2 分钟, 800 tokens
  → Agent: "我发现你每次部署前都运行测试，要加入自动化吗？"
```

三层进化：
1. **自动技能生成**: 同类任务成功 ≥3 次 → 自动创建 SKILL.md
2. **GEPA 进化引擎**: 帕累托多目标优化（准确率↑ + 成本↓ + 延迟↓）
3. **预测记忆**: 学习你的习惯 → 在你开口前准备好上下文

---

## 记忆系统 — 4 层 + 主动学习

| 层级 | 功能 | 增强 |
|------|------|------|
| L1 冻结 | 结构化日报（时间戳、文件列表、发现） | 未完成任务(○)保留，已完成自动压缩 |
| L2 程序性 | 自动技能 + 10 种角色专属技能 | 角色检测自动加载对应技能 |
| L3 FTS5 索引 | 全文搜索 + LLM 语义扩展 | 智能回退：静态搜索优先(0ms)，无结果时 LLM 增强 |
| L4 预测 | 行为预测 | — |

**新增——主动学习管道**：
- 每次任务 → 结构化日报 + 教训提取
- 重复错误 → 自动防御规则
- 新会话 → 注入最近上下文 + 相关历史教训
- 项目文件清单跨会话持久化

---

## 问题解决增强系统

DeepSoul 内置了 9 个推理增强模块，全部零额外 LLM 成本：

| 模块 | 功能 |
|------|------|
| 执行前规划 | LLM 先输出 JSON 计划再执行，追踪进度 |
| 侦察阶段 | 动手前用只读工具快速摸底现状 |
| 模糊反问 | 任务太模糊时反问 1-2 个关键问题 |
| 自纠错闭环 | 失败→诊断→修正→重试，连续 2 次失败切换方案 |
| 结果验证 | 每次工具执行后自动验证输出质量（18 种错误检测） |
| 编译检查 | 修改代码后强制运行编译/语法检查 |
| 工作记忆 | 追踪已尝试方法、排除方向、错误模式 |
| 错误知识库 | 跨会话累积修复方案，越用越准 |
| 断点续跑 | 长任务崩溃后从断点恢复，不丢失进度 |

---

## 架构

```
deepseek-SOUL/
├── soul/                    # 核心引擎 (~9600 行)
│   ├── engine/              # Agent 循环 + Lane Queue 2.0 并发调度 + 6 个增强模块
│   ├── memory/              # 4 层记忆系统 + 错误知识库
│   ├── skills/              # 自动技能生成 + GEPA 进化引擎
│   ├── tools/               # 工具系统 + 4 层安全护栏
│   ├── prompt/              # Prompt 构建 + 前缀缓存 + 上下文压缩
│   ├── llm/                 # DeepSeek / Claude / OpenAI 适配器
│   ├── gateway/             # 统一消息网关 + WebSocket + 5 平台连接器
│   ├── safety/              # Docker 沙箱 + DM 配对 + 审计
│   ├── cron/                # 自然语言定时任务
│   ├── mlops/               # 轨迹生成 → 压缩 → LLM 评估
│   └── config/              # 统一配置管理 (YAML + 环境变量)
├── skills/bundled/          # 25 个内置技能
├── web/                     # FastAPI + WebSocket Web 界面
├── scripts/install.sh       # 一行安装
└── pyproject.toml
```

---

## CLI 参考

```bash
deepsoul chat                   # 交互式对话
deepsoul chat "写个脚本"         # 单条消息（非交互）
deepsoul run "创建项目"          # 执行单次任务

deepsoul gateway                 # 启动网关 (端口 18789)
deepsoul config --all            # 查看所有配置
deepsoul config llm.model gpt-4o # 修改单项配置
deepsoul status                  # 系统状态
deepsoul doctor                  # 诊断 (依赖/配置/API Key)

deepsoul train tasks.txt --count 1000 --workers 8  # 生成训练数据
deepsoul version                 # 版本信息
```

---

## API 使用

```python
from soul.engine.agent import Agent
from soul.config.manager import ConfigManager

config = ConfigManager().load()
agent = Agent(config=config)
await agent.initialize()

# 对话
response = await agent.chat("创建一个 Python 项目")
print(response)

# 流式
async for chunk in agent.chat_stream("分析这个代码库"):
    print(chunk.content, end="")
```

HTTP 方式：
```bash
curl -X POST http://localhost:18789/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "写一个快速排序实现"}'
```

---

## 环境要求

- Python 3.11+
- DeepSeek / Claude / OpenAI API key
- （可选）Docker 用于沙箱模式
- （可选）ripgrep 加速代码搜索

---

## 许可证

MIT License — 自由使用，自由修改。
