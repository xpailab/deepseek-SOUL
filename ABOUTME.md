# DeepSoul — 工作机制与设计哲学

## 一句话定位

DeepSoul 是一个**会进化的 AI Agent 框架**。融合 OpenClaw 的编排能力（370K★）与 Hermes Agent 的自我进化（148K★），用约 1/3 的代码量实现两者的核心能力。

## 核心哲学

```
OpenClaw = 连接一切的"手"（编排 + 并发 + 多平台）
Hermes   = 陪你成长的"脑"（记忆 + 技能 + 进化）
DeepSoul = "手" + "脑" = 既会干，又会学
```

---

## 总体架构

```
用户输入
  ↓
① Lane Queue 2.0（并发调度）
  ↓
② Agent 核心循环
  ↓
③ Prompt 组装（7层注入 + 冻结快照）
  ↓
④ LLM 推理（DeepSeek/Claude/OpenAI 统一适配）
  ↓
⑤ 工具调用管道（4层安全护栏）
  ↓
⑥ 记忆更新（4层并行写入）
  ↓
⑦ 技能进化（GEPA 后台异步）
  ↓
回复输出
```

---

## ① Lane Queue 2.0 — 并发调度机制

这是从 OpenClaw 学的最核心工程实现。所有用户消息都先经过这个双层队列：

```
消息到达
    ↓
Session Lane（per session，并发 = 1）
    ├── 同一会话的消息强制串行
    └── 保证 session 内上下文不乱序
        ↓
Global Lane（全局，并发 = 4）
    ├── asyncio.Semaphore(4) 控制
    ├── 获取槽位 → 执行 → 释放槽位
    └── 防止并发过多打爆 LLM API
        ↓
Worker → Agent 实例执行
```

### 7 种队列模式

| 模式 | 行为 | 自动触发条件 |
|------|------|-------------|
| `interrupt` | 立即中断当前运行 | 优先级 ≥ 8 |
| `steer` | 立即注入当前流式输出 | 优先级 ≥ 5 + 正在流式 |
| `steer_backlog` | 注入当前 + 排队后续处理 | 正在流式 + 优先级 ≥ 3 |
| `collect` | 合并积压消息为一条 | 会话忙 + 全局积压 > 5 |
| `followup` | 排队等下一回合 | 会话空闲 |
| `queue` | 标准 FIFO | 默认 |
| `adaptive` | 智能自动选择以上模式 | **SOUL 创新** |

`adaptive` 模式根据消息优先级 × Agent 状态 × 系统负载自动决策，消除了手动选择模式的心智负担。

---

## ② Agent 核心循环

代码在 `soul/engine/agent.py`。一次完整的对话过程：

```
1. 入队（Lane Queue）
   → 消息被封装为 QueueItem → 入队到 Session Lane → 竞争 Global Lane 槽位

2. 出队获取执行权限
   → 取得 Global Lane 信号量 → 开始执行

3. 记忆检索（4 层并行）
   → 查询 4 层记忆 → 返回相关上下文 → 注入 system prompt

4. 组装 system prompt（7 层注入）
   → SOUL.md + IDENTITY.md + AGENTS.md + USER.md + MEMORY.md
   + 匹配技能 + 工具声明 + 安全护栏

5. LLM 推理
   → 调用 DeepSeek/Claude/OpenAI API → 返回文本 + 工具调用

6. 工具调用管道
   → 参数校验 → 路径沙箱 → 命令审批 → 执行 + 重试 → 结果分类
   → 结果反馈给 LLM → LLM 继续推理或生成最终回复

7. 记忆更新
   → 对话历史写入 FTS5
   → Agent 笔记更新 MEMORY.md
   → 行为模式喂入预测引擎
   → 成功任务触发技能生成

8. 释放槽位
   → Global Lane 信号量 +1 → 下一个消息可以执行
```

### 流式输出的特殊处理

```python
# chat_stream 中注册 steer 回调
self.lane_queue.track_active(session_id)  # 标记为活跃，允许 steer 注入
try:
    async for chunk in self.llm.chat_stream(...):
        # 排出 steer 队列
        while not steer_queue.empty():
            yield StreamChunk("[注入: 用户的即时修正]")
        yield chunk
finally:
    self.lane_queue.untrack_active(session_id)  # 取消标记
    self.lane_queue.mark_done(session_id)        # 释放槽位
```

---

## ③ Prompt 组装 — 7 层注入 + 冻结快照

### 七层内容

```
最终 System Prompt 由这些层叠起来：
  ┌──────────────────────────┐
  │ ① SOUL.md      人格定义   │ 性格/边界/风格
  │ ② IDENTITY.md  Agent身份 │
  │ ③ AGENTS.md    行为说明   │
  │ ④ USER.md      用户画像   │ 偏好/习惯/常用工具
  │ ⑤ MEMORY.md    Agent笔记  │ 项目上下文/经验教训
  │ ⑥ SKILL.md     匹配技能   │ 语义匹配召回的相关技能
  │ ⑦ 工具声明     API schema │ 所有可用工具的 JSON Schema
  │ ⑧ 安全护栏     自动注入   │ 全局行为约束规则
  └──────────────────────────┘
```

### 冻结快照机制（保护 LLM prefix cache）

```
会话 N 开始
  → 读取 MEMORY.md v1 → 计算哈希 SHA256 → 冻结快照 → 注入 prompt
  → LLM 处理时生成 prefix cache（prompt 前缀不变 → cache 持续命中）
  → Agent 执行中修改 MEMORY.md → 只写磁盘 → 当前 prompt 不变
  → 本轮对话的 prefix cache 始终有效

会话 N+1 开始
  → 读取 MEMORY.md v2 → 哈希变了 → 冻结新快照 → 注入新 prompt
  → prefix cache 自动失效 → LLM 重新处理一次 → 再缓存
```

**目的**：避免每次对话都让 LLM 重新处理整个 system prompt。如果 prompt 频繁变化，cache 不断失效，推理成本和延迟显著增加。冻结快照确保单会话内 prompt 不变，最大化缓存命中率。

**代价**：记忆更新延迟一个会话周期才生效。

---

## ④ LLM 推理 — 多提供商统一适配

```
BaseAdapter（统一接口）
  ├── chat(messages, tools, system_prompt) → LLMResponse
  ├── chat_stream(...) → AsyncIterator[StreamChunk]
  ├── supports_tools() → bool
  └── count_tokens(text) → int

  ├── DeepSeekAdapter  → POST https://api.deepseek.com/v1/chat/completions
  ├── ClaudeAdapter    → POST https://api.anthropic.com/v1/messages
  └── OpenAIAdapter    → POST https://api.openai.com/v1/chat/completions
```

每个适配器都处理：
- 内部 `Message` 格式 ↔ 各 API 原生格式的转换
- 非流式（`chat`）和流式（`chat_stream`）两种模式
- 工具调用格式的转换（内部 `ToolCall` ↔ OpenAI function calling / Claude tool use）
- 指数退避重试：1s → 2s → 4s → 最多 3 次
- 令牌桶速率限制（默认 60 requests/min）

---

## ⑤ 工具调用管道 — 四层安全护栏

一次 `bash "rm -rf /tmp/cache"` 调用的完整链路：

```
LLM 产生 tool_call:
  {name: "bash", arguments: {command: "rm -rf /tmp/cache"}}
  ↓
第 1 层：参数校验（guardrails.py）
  ├── 扫描 prompt injection 标签
  ├── 按工具名匹配检查规则
  │   ├── bash → 正则匹配危险模式
  │   │   rm -rf /, mkfs., dd, curl | bash, sudo, chmod 777 /
  │   │   fork bomb :(){ :|: };, eval $...
  │   └── file → 路径必须在工作空间内（Path.relative_to 检查）
  ├── 拦截 → 返回 denied
  └── 通过 → 继续
  ↓
第 2 层：审批门禁（agent.py）
  ├── risk=CRITICAL → 返回错误，需用户确认
  └── 当前实现：高风险操作直接拒绝
  ↓
第 3 层：执行 + 重试（retry.py）
  ├── 速率限制检查（令牌桶，滑动窗口 60 req/min）
  ├── 执行工具 handler
  ├── 成功 → 返回 result
  ├── Timeout → 指数退避 1s→2s→4s 重试最多 3 次
  ├── PermissionError → 不重试，直接返回 denied
  ├── Rate Limit → 等待后重试
  └── 异常捕获 → 返回 error
  ↓
第 4 层：结果分类（classifier.py）
  ├── success → 直接返回
  ├── partial → 标记部分成功
  ├── denied → 提示用户授权
  ├── failure → 错误分析 + 根因诊断
  ├── timeout → 中断 + 结果摘要
  └── rate_limited → 等待 + 重试

最终：ToolResult 对象（含分类）返回 Agent 循环
  → LLM 根据工具结果继续推理或生成最终回复
```

---

## ⑥ 四层记忆系统

代码在 `soul/memory/manager.py`，统一调度四层记忆。

```
Layer 1: FrozenMemory（冻结快照）
  存储：MEMORY.md ≤ 2,200 chars、USER.md ≤ 1,375 chars
  写入：Agent 调用 memory.remember("内容")
        → 直接写磁盘文件
        → 当前 session 的 prompt 不更新（保护 prefix cache）
  读取：下个 session 开始时自动加载
  容量控制：超过 80% 自动压缩，§ 分隔条目，按顺序保留

Layer 2: ProceduralMemory（程序技能）
  存储：~/.soul/skills/*.skill 文件
  写入：任务成功 3 次
        → SkillGenerator.analyze_trajectory(轨迹)
        → 提取步骤模式 + 使用的工具 + 关键决策点
        → 生成 SKILL.md（遵循 agentskills.io 标准）
  读取：语义匹配
        → 关键词匹配（名称/描述/触发词/内容）
        → 成功率加权 × 使用频率加权 × fitness_score 加权
        → top-k 召回 → 注入 system prompt

Layer 3: IndexedMemory（FTS5 + LLM 混合检索）
  存储：SQLite FTS5 表
        conversations(id, session_id, role, content, timestamp)
        conversation_fts(content, role, session_id)  -- 全文索引
  写入：每条消息 INSERT 进 conversations + FTS 索引
  读取：FTS5 MATCH 全文搜索
        → 转义特殊字符防注入
        → 关键词扩展（部署→deploy/上线/发布/release）
        → FTS5 精确召回（人名/项目名/命令不丢）
        → LLM 语义理解补充摘要
  设计哲学：零运维，$5 VPS 即可，不依赖向量数据库
  补偿方案：LLM 摘要层补偿 FTS5 的语义短板

Layer 4: PredictiveMemory（预测记忆）★ SOUL 创新
  存储：内存中的行为图谱
        _task_graph: 马尔可夫转移概率（做完A→80%概率做B）
        _temporal_index: 时间段关联（每天早上9点做的事）
        _context_index: 项目上下文关联（在哪个项目做什么）
        _habits: 习惯计数器（同一件事重复了多少次）
  写入：每次 observe_action(current, previous, context)
        → EMA 更新转移概率 → 归一化
        → 时间段索引更新 → 上下文索引更新 → 习惯计数 +1
  读取：
        predict_next_actions → 路径概率 × 时间关联 × 上下文关联
                           → top-3 预测 → 注入 <predictive_context>
        detect_habits → 计数器 ≥ 3 的习惯 → 生成自动化建议
                      → "你每次改完代码都跑测试，要设置成自动执行吗？"
```

### 四层 vs 三层的本质区别

Hermes 的记忆系统是被动响应的——你问什么，它搜什么。

DeepSoul 的第 4 层预测记忆改变了这个范式：**在你开口之前，就已经准备好了你可能需要的上下文**。

---

## ⑦ GEPA 技能进化引擎

算法来自 Hermes Agent 的 ICLR 2026 Oral 论文。代码在 `soul/skills/gepa.py`。

```
初始化
  ├── 加载当前技能版本为基线（generation = 0）
  └── 生成 population_size = 8 个随机文本变体
       ↓
迭代（最多 10 代）
  ├── 变异（mutation_rate = 0.3）
  │   ├── 分析执行失败追踪
  │   │   ├── "permission denied" → 添加权限检查步骤
  │   │   ├── "timeout" → 添加超时处理和重试逻辑
  │   │   └── "not found" → 添加文件/资源存在性检查
  │   └── 无追踪 → 随机变异（重组步骤/添加示例/补充说明）
  │
  ├── 交叉（crossover_rate = 0.5）
  │   └── 两个父代在段落边界处交换内容（取 A 的前半 + B 的后半）
  │
  ├── 评估（并行）
  │   ├── accuracy（50%）：内容结构完整性、步骤明确性、长度适中
  │   ├── cost（25%）：文本长度 → 越短越好
  │   └── latency（25%）：步骤复杂度 → 越少越好
  │   综合适应度 = 0.5×accuracy + 0.25×cost + 0.25×latency
  │
  ├── 选择（帕累托多目标优化）
  │   ├── 精英保留 top-2（总体最高分）
  │   ├── 各维度最优（accuracy/cost/latency 各自第一名）
  │   └── 填充剩余位置（按综合分排序）
  │
  └── 早停：overall ≥ 0.95 或达到 max_iterations = 10
       ↓
输出：进化后的技能
  ├── gepa_generation = 实际代数
  ├── fitness_score = 0.85 → 0.98
  ├── version: 1.0.0 → 1.0.5
  └── content 包含了针对性改进步骤
```

**关键指标**：
- 纯 API 调用，无需 GPU
- 每次优化成本 $2-10
- 仅需 3 个示例即可启动
- 优化对象是**提示文本**而非模型权重
- 安全护栏：技能文件 ≤ 15KB，工具描述 ≤ 500 chars

---

## 完整流程示例

```
用户: "帮我创建一个 FastAPI 项目，包含 JWT 认证"

① Lane Queue:
   → 消息优先级 0（普通）
   → ADAPTIVE 模式 → 会话空闲 → FOLLOWUP
   → 获取 Global Lane 槽位（Semaphore 4→3）

② Agent 开始:
   → session_id = "sess_a1b2c3d4e5f6"
   → 状态: IDLE → THINKING

③ Prompt 组装:
   → SOUL.md: "你是一个高效专业的 AI 助手..."
   → USER.md: "用户是 Python 开发者，偏好 FastAPI..."
   → MEMORY.md: "上次创建的项目用了 PostgreSQL + SQLAlchemy..."
   → 匹配技能: python_dev.skill（触发词: python + 项目 + 创建）
   → 工具声明: [bash, file, web]
   → FTS5 检索: 3 天前的对话 "FastAPI JWT 最佳实践"
   → 预测记忆: 空（首次此类任务）

④ LLM 推理:
   → POST DeepSeek API
   → 返回:
     content: "好的，我来创建项目结构..."
     tool_calls: [
       {name: "file", args: {action: "mkdir", file_path: "myapi"}},
       {name: "file", args: {action: "write",
         file_path: "myapi/main.py",
         content: "from fastapi import FastAPI\n..."}},
       ...共 8 次工具调用
     ]

⑤ 工具调用管道:
   → mkdir("myapi"): 工作空间内 ✓ → 执行 → success
   → write("myapi/main.py"): 工作空间内 ✓ → 执行 → success
   → write("myapi/auth.py"): 工作空间内 ✓ → 执行 → success
   → ... 全部 8 次调用成功

⑥ 记忆更新:
   → L1: MEMORY.md += "创建了 FastAPI 项目 myapi 于 {时间}"
   → L2: 尚需 2 次成功才会自动生成技能
   → L3: 完整对话写入 FTS5，可全文检索
   → L4: 记录 (创建FastAPI项目) 的行为模式

⑦ 回复用户:
   → 释放 Global Lane 槽位（Semaphore 3→4）
   → 返回完整的项目代码和说明
```

---

## 安全架构

```
数据隐私
  ├── 所有数据存储在 ~/.soul/（本地）
  ├── 零遥测（无数据外传）
  └── 可审计（auditor.py 记录所有敏感操作）

执行安全
  ├── 沙箱隔离
  │   ├── Local：asyncio subprocess + 环境变量清理
  │   ├── Docker：只读根文件系统 + 内存/CPU 限制 + 网络隔离
  │   └── SSH：shlex.quote 转义防注入
  ├── 路径隔离
  │   ├── 所有文件操作强制在工作空间内
  │   └── blocked_paths: /etc/passwd, /etc/shadow, ~/.ssh
  └── 命令过滤
      ├── 正则匹配危险模式（rm -rf /, mkfs., dd, fork bomb...）
      ├── 高风险工具（bash）需通过防护检查
      └── CRITICAL 风险操作直接拒绝

交互安全
  ├── DM 配对：未知发送者需配对码
  ├── Prompt injection 扫描：检测 <system_reminder>、角色扮演等模式
  └── 敏感信息脱敏：API key 在日志中自动截断
```

---

## 部署与运维

```
安装
  curl | bash 一键安装
  ├── 自动检测系统（Linux/macOS/WSL2）
  ├── 自动安装 Python 3.11+（如缺失）
  ├── 自动创建虚拟环境 + 安装依赖
  ├── 交互式配置 API Key + 模型
  └── 自动添加到 PATH

运行
  soul chat          → 交互式对话
  soul run "任务"     → 单次执行
  soul gateway       → 启动网关（REST + WebSocket + Web UI）
  soul doctor        → 系统诊断
  soul train         → MLOps 训练管道

容器化
  docker build -t deepseek-soul .
  docker run -v ~/.soul:/root/.soul deepseek-soul chat
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| Python 而非 TypeScript | AI/ML 生态更好，部署更简单 |
| Pydantic v2 | 全类型安全，配置校验，零运行时意外 |
| asyncio 而非多线程 | I/O 密集（LLM API 调用），asyncio 最合适 |
| SQLite FTS5 而非向量数据库 | 零运维，$5 VPS，精确匹配不丢（人名/项目名） |
| 冻结快照而非实时更新 prompt | 保护 LLM prefix cache，降低 API 成本 |
| 文件化技能而非数据库 | 人类可读，社区可分享，Git 可追踪 |
| 单配置文件 | 降低认知负担，一份文件覆盖全部功能 |

---

## 消息处理完整调用链

以用户输入 `"帮我打开豆包"` 为例，逐行追踪代码执行路径：

```
cli.py:104    console.input("You> ")              ← 用户输入消息
cli.py:127    agent.chat_stream(user_input, sid)  ← 调 Agent 流式接口
      │
      ▼
agent.py:271  async def chat_stream(              ← 入口
agent.py:274  sessions.get_or_create(sid)         ← 查找/创建会话
agent.py:275  lane_queue.enqueue(item)            ← 消息入队
lane_queue.py:307  item={prompt="打开豆包", mode=ADAPTIVE}
lane_queue.py:341  ADAPTIVE → 会话空闲 → FOLLOWUP
agent.py:295  lane_queue.dequeue(sid)             ← 出队取执行槽位
lane_queue.py:367  Global Lane Semaphore(4→3)
agent.py:304  track_active(sid)                   ← 标记流式活跃
      │
agent.py:308  sessions.get_history(sid)           ← L0: 查会话历史
session.py:95
      │
agent.py:311  memory.query_for_prompt("打开豆包") ← L2+L3+L4 检索
memory/manager.py:120
      ├─ L2: procedural.match("打开豆包")        ← 技能匹配
      ├─ L3: indexed.search_semantic("打开豆包")  ← FTS5 全文搜索
      └─ L4: predictive.get_predictive_context()  ← 预测上下文
      │
agent.py:314  prompt_builder.build_system_prompt( ← 组装 system prompt
prompt/builder.py:68
      ├─ ① SOUL.md          (人格定义)
      ├─ ② IDENTITY.md      (Agent 身份)
      ├─ ③ AGENTS.md        (行为说明)
      ├─ ④ USER.md          (用户画像)
      ├─ ⑤ MEMORY.md        (Agent 笔记)
      ├─ ⑥ matched_skills   (语义匹配的技能内容)
      ├─ ⑦ tools schema     (bash/file/web 的 JSON Schema)
      └─ ⑧ safety+global    (安全护栏 + 全局行为规则)
      │
agent.py:350  llm.chat_stream(current_messages)   ← 调用 DeepSeek API
deepseek.py:127  POST https://api.deepseek.com/v1/chat/completions
deepseek.py:161  messages → _messages_to_api_format()
llm/base.py:87    转换: system + user + assistant+tc + tool → API格式
      │
      ▼  LLM 返回: content="我来找" + tool_calls=[bash: ls Desktop]
      │
agent.py:369  ── 工具调用循环 R1 ──
agent.py:395    for tc in tool_calls:
agent.py:441      _execute_tool(tc)
                    ├─ guardrails.py:86  check_tool_call("bash", {cmd})
                    │  guardrails.py:158   _check_command → 19条正则检查 → OK
                    ├─ requires_approval? → False
                    ├─ retry.py:76        execute_with_retry(bash_handler)
                    │  → create_subprocess_shell("ls ~/Desktop")
                    │  → stdout bytes → bash.py:_decode_output → 系统编码解码中文
                    └─ classifier.py:58   classify("bash", result)
                       → ToolResult(success, stdout="豆包.lnk")
agent.py:401    assistant_msg → current_messages (保留 reasoning_content)
agent.py:410    tool_msg(TOOL角色) → current_messages (独立 role=tool)
agent.py:350  ── LLM 再调 R2 ──
              当前 messages: [system, user, assistant+tc, tool(result)]
              返回: "找到了，打开它" + tool_calls=[bash: start 豆包.lnk]
      │
agent.py:369  ── 工具调用循环 R2 ──
agent.py:395    _execute_tool(bash: start 豆包.lnk)
                → 豆包进程启动成功
agent.py:350  ── LLM 再调 R3 ──
              返回: "豆包已启动!" + tool_calls=[]
agent.py:366    finish_reason="stop" + 无工具调用 → break
      │
      ▼
agent.py:423  ── 记忆写回 ──
agent.py:429    memory.observe_action("打开豆包")        → L4: 更新行为图谱
memory/manager.py:218
agent.py:430    memory.store_conversation(sid, messages)  → L3: 写入 FTS5
memory/manager.py:196
agent.py:432    memory.remember("用户:打开豆包|回复:已启动") → L1: 写 MEMORY.md
memory/manager.py:163
      │
agent.py:437  finally: untrack_active(sid)
agent.py:439    lane_queue.mark_done(sid) → Global Lane Semaphore(3→4)
lane_queue.py:388
      │
cli.py:129    console.print("豆包已启动!")              ← 用户看到回复
```

**共 13 个源文件参与处理，每个箭头标注了精确的文件名和行号。**

### 被截断时的自动续写

当 LLM 返回 `finish_reason="length"`（token 不够被截断）时：

```
agent.py:371    检测到 finish_reason == "length"
agent.py:372    → 不 break，自动发系统消息 "请继续。"
agent.py:350    → 下一轮 LLM 调用，续写被截断的内容
                → 所有轮次的 content 用 += 累积拼接
```

### 异常保护

```
agent.py:373    finish_reason == "error" → await asyncio.sleep(1.5) → continue 重试
agent.py:425    final_content 为空 → 返回 "抱歉，任务未能完成。请重试或简化您的请求。"
```

---

## 记忆系统读写全链路

### 存储文件

```
C:\Users\<用户>\.soul\
├── SOUL.md                    ← Agent 人格定义
├── predictive.json            ← L4 预测模型持久化
├── memory.db                  ← L3 SQLite FTS5 对话索引
├── skills/                    ← L2 用户技能 (空)
└── workspace/
    ├── MEMORY.md              ← L1 Agent 笔记
    ├── USER.md                ← L1 用户画像
    └── sessions/              ← 会话历史 JSON
```

### 启动时一次性读

```
agent.py:108  Agent.initialize()
                │
    ┌───────────┼───────────┬───────────────┐
    ▼           ▼           ▼               ▼
manager:51   manager:53  manager:54     (L3 延迟建表)
L2 加载       L1 快照     L4 恢复        首次查询时 CREATE TABLE
从磁盘读:    从磁盘读:    从磁盘读:
 skills/    MEMORY.md  predictive.json → 内存
 bunded/    USER.md    _task_graph, _habits,
 → 内存      → 内存冻结  _temporal_index
```

### 每条消息 — 读阶段（构建 system prompt 前）

```
agent.py:311  memory.query_for_prompt(user_message)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
manager:118  manager:127  manager:136
L2 match()  L3 search()  L4 detect()
查内存索引   查 SQLite    查内存图谱
→ 技能列表   → 历史对话   → 习惯+预测
→ prompt    → prompt    → prompt

agent.py:314  prompt_builder.build_system_prompt()
                │
builder.py:72 读 SOUL.md      → <soul_personality>
builder.py:76 读 IDENTITY.md  → <agent_identity>
builder.py:80 读 AGENTS.md    → <agent_behavior>
builder.py:84 读 USER.md      → <user_profile>
builder.py:88 读 MEMORY.md    → <agent_memory>
builder.py:99 技能列表        → <available_skills>
builder.py:103 工具声明        → <available_tools>
builder.py:113 安全护栏        → <safety_rules>
builder.py:116 全局规则        → <global_rules>
```

### 每条消息 — 写阶段（对话结束后）

```
agent.py:429  memory.observe_action(user_message)
                │
manager.py:218  → predictive.observe()
                  → 更新内存 _task_graph (马尔可夫转移概率)
                  → 更新内存 _temporal_index (时间段索引)
                  → 更新内存 _habits (习惯计数器)
                  不写磁盘！

agent.py:430  memory.store_conversation(sid, messages)
                │
manager.py:196  → indexed.store_message() × N 条
                  → BEGIN IMMEDIATE 事务
                  → INSERT INTO conversations (...)
                  → INSERT INTO conversation_fts (...)
                  → COMMIT
                  写 ~/.soul/memory.db

agent.py:431  memory.remember(summary, FROZEN)
                │
manager.py:171  → frozen.add("MEMORY.md", content)
                  → 读当前 MEMORY.md → 追加 "§ 新内容"
                  → filepath.write_text(new_content)
                  写 ~/.soul/workspace/MEMORY.md
                  当前会话 prompt 不变（冻结快照保护 prefix cache）
```

### 关闭时写

```
agent.py:517  Agent.shutdown()
                │
manager.py:251  predictive.save()
                  → json.dumps(_task_graph, _habits, _temporal, _context)
                  → 写 ~/.soul/predictive.json

manager.py:252  indexed.close()
                  → 关闭 ~/.soul/memory.db 连接
```

### 四层汇总

| 层 | 存储位置 | 读触发 | 写触发 | 注入位置 |
|----|---------|--------|--------|---------|
| L1 冻结 | `MEMORY.md` `USER.md` `SOUL.md` | 启动 + 每条消息 | 每条消息 `remember()` | `<agent_memory>` `<user_profile>` `<soul_personality>` |
| L2 技能 | `~/.soul/skills/*.skill` `skills/bundled/*.skill` | 启动 + 每条消息 `match()` | 未触发 (`learn_skill`) | `<available_skills>` |
| L3 检索 | `~/.soul/memory.db` (SQLite FTS5) | 每条消息 `search_semantic()` | 每条消息 `store_conversation()` | `<context>` |
| L4 预测 | `~/.soul/predictive.json` (内存+JSON) | 启动 `load()` + 每条消息 | 每次 `observe_action()` + 关闭 `save()` | `<predictive_context>` |

### 冻结快照机制

```
会话 N 开始
  → 读 MEMORY.md v1 → SHA256 → 冻结快照
  → system prompt 注入 v1
  → Agent 执行中 remember() → 写磁盘 v2 → prompt 仍为 v1
  → LLM prefix cache 持续命中

会话 N+1 开始
  → 读 MEMORY.md v2 → SHA256 变了 → 冻结新快照
  → system prompt 注入 v2 → cache 重置一次
```

---

## 许可证

MIT License
