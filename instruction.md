# DeepSoul 内部处理与流转路径

> 基于 `soul/engine/agent.py` 等源码逐行确认，非文档推测。

---

## 一、代码层面：模块调用链

### 阶段 0：启动与初始化

```
cli.py:69   agent = Agent(config=config)
             ├─ agent.py:125-178  __init__ 创建全部子系统:
             │   self.llm           = AdapterRegistry()        ← LLM 适配器注册表
             │   self.lane_queue    = LaneQueue(config.lane)    ← 双层并发调度
             │   self.sessions      = SessionManager(...)      ← 会话状态管理
             │   self.memory        = MemoryManager(config)    ← 4 层记忆
             │   self.prompt_builder= PromptBuilder(...)       ← 系统提示组装
             │   self.compressor    = ContextCompressor()      ← 上下文压缩
             │   self.tools         = ToolRegistry()           ← 工具注册表
             │   self.guardrails    = ToolGuardrails(...)      ← 安全护栏
             │   self.classifier    = ResultClassifier()       ← 结果分类器
             │   self.retry_mgr     = RetryManager()           ← 重试管理器
             │   self.sandbox       = Sandbox(...)             ← 执行沙箱
             │   self.auditor       = Auditor()                ← 审计记录
             │   self.startup_cwd   = os.getcwd()             ← 启动时 CWD
             │   self.working_memory= WorkingMemory()          ← 会话级工作记忆
             │   self.verifier      = ResultVerifier()         ← 输出验证器
             │   self.error_kb      = ErrorKnowledgeBase()     ← 跨会话错误知识库
             │   self.checkpoint_mgr= CheckpointManager()      ← 断点续跑

cli.py:72   await agent.initialize()
             ├─ agent.py:181  _register_builtin_tools()
             │   → BashTool / FileTool / WebTool / BrowserTool / WindowsTool
             ├─ agent.py:184  memory.initialize()
             │   → 加载 4 层记忆 + bundled skills
             ├─ agent.py:187  sessions.get_or_create("main")
             └─ agent.py:190  error_kb.load()
```

### 阶段 1：消息入队（Lane Queue）

```
agent.py:199  async def chat(user_message, session_id, system_prompt, tools):
                │
                ├─ agent.py:210  session = sessions.get_or_create(session_id)
                │
                ├─ agent.py:214  item = QueueItem(
                │                    id=f"msg_{timestamp}", session_id,
                │                    prompt=user_message, mode=QueueMode.ADAPTIVE
                │                )
                │
                ├─ agent.py:222  result = lane_queue.enqueue(item)
                │   └─ lane_queue.py:307  enqueue()
                │       ├─ session_lane = global_lane.get_or_create_session_lane(sid)
                │       ├─ mode = resolve_mode(item, is_streaming, is_busy)
                │       │   └─ ADAPTIVE → _adaptive_choice():
                │       │       优先级≥8→INTERRUPT | 流式+≥5→STEER
                │       │       繁忙+积压→COLLECT  | 空闲→FOLLOWUP | 默认→QUEUE
                │       └─ 按 mode 分派: steer/collect/enqueue
                │
                ├─ agent.py:227  item = lane_queue.dequeue(session_id)
                │   └─ lane_queue.py:367  dequeue()
                │       ├─ acquired = global_lane.acquire()   ← Semaphore(4)
                │       └─ item = session_lane.dequeue()      ← asyncio.Queue
                │
                └─ agent.py:234  sessions.update_state(sid, THINKING)
```

### 阶段 2：记忆检索 + Prompt 构建

```
agent.py:237  history = sessions.get_history(session_id)

agent.py:240  memory_context = memory.query_for_prompt(user_message)
              └─ memory/manager.py:136  query_for_prompt(query)
                  ├─ L2: procedural.match(query, top_k=2)     → 关键词匹配技能
                  ├─ L3: indexed.search_semantic(query, limit=3) → FTS5 搜索历史
                  ├─ L4: predictive.get_predictive_context_prompt() → 行为预测
                  └─ UserModel: get_full_prompt_fragment()    → 用户画像

agent.py:241  matched_skills = memory.procedural.match(user_message, top_k=2)

agent.py:242  base_system_prompt = prompt_builder.build_system_prompt(
                  matched_skills, tools, extra_context=memory_context
              )
              └─ prompt/builder.py:69  build_system_prompt()
                  按序组装 11 段 XML:
                  ① <soul_personality>     ← SOUL.md
                  ② <agent_identity>      ← IDENTITY.md
                  ③ <agent_behavior>      ← AGENTS.md
                  ④ <user_profile>        ← USER.md
                  ⑤ <agent_memory>        ← MEMORY.md
                  ⑥ <tools_guide>         ← TOOLS.md
                  ⑦ <available_skills>    ← 匹配到的技能（如有）
                  ⑧ <available_tools>     ← 工具声明（如有）
                  ⑨ <context>             ← 记忆检索结果（如有）
                  ⑩ <safety_rules>        ← 硬编码安全规则
                  ⑪ <global_rules>        ← 硬编码行为规则
                      含: 侦察/反问/编辑策略/编码验证/自纠错/执行报告规则
                  所有用户文本经 _sanitize() 转义危险标签
```

### 阶段 3：增强注入

```
agent.py:251  enhanced_prompt = self._build_enhanced_prompt(
                  base_system_prompt, user_message, first_round=True
              )
              └─ agent.py:875  _build_enhanced_prompt()
                  parts = [base_prompt]
                  │
                  ├─ ① 工作记忆: wm.to_prompt()
                  │     → 计划进度 / 已尝试方法 / 排除方向 / 发现 / 错误 / 验证失败
                  │
                  ├─ ② 首轮 (first_round=True):
                  │   ├─ checkpoint_mgr.load_latest(max_age_hours=1) → 续跑或跳过
                  │   ├─ _recon_prompt(user_message)
                  │   │   → CWD + "先侦察再动手" + 模糊反问检测
                  │   ├─ _coding_cadence_prompt() → "小步快跑"编码节拍
                  │   └─ wm.get_planning_prompt() → JSON 计划模板
                  │
                  └─ ③ 非首轮 (first_round=False):
                      ├─ needs_full_rewrite() → 逐行修补死循环检测
                      ├─ _coding_guard_from_memory() → 编译检查注入
                      ├─ _regression_guard() → 全量回归检查
                      └─ error_kb.lookup_by_confidence() → 已知修复建议

agent.py:254  self.working_memory.clear()
agent.py:257  user_msg = Message(role=USER, content=user_message)
agent.py:259  full_messages = prompt_builder.build_messages(history+[user_msg], ...)
```

### 阶段 4：LLM 推理循环（max 50 轮）

```
agent.py:276  for round_num in range(max_rounds):  ← 前有 max_rounds=50 等变量初始化

  ┌─ 停止条件 ─────────────────────────────────────
  │ if len(all_tool_results) >= 100:  break   ← 100次工具调用上限
  │ if consecutive_fails >= 5:        break   ← 连续5次失败中止
  │
  ├─ 上下文压缩（每5轮）───────────────────────────
  │ compressor.needs_compression() → compress()
  │   策略: 滑动窗口 → 中间摘要 → LLM 智能压缩
  │
  ├─ 每轮重建 live prompt ────────────────────────
  │ live_prompt = _build_enhanced_prompt(base, msg, first_round=False)
  │
  ├─ LLM 调用 ────────────────────────────────────
  │ response = self.llm.chat(messages, tools, system_prompt=live_prompt, ...)
  │   └─ llm/registry.py:88  AdapterRegistry.chat()
  │       ├─ adapter = get(provider)     ← 按 "provider:model" 缓存
  │       ├─ rl = _get_rate_limiter()    ← 令牌桶限流
  │       └─ adapter.chat()
  │           └─ llm/deepseek.py:48  DeepSeekAdapter.chat()
  │               ├─ POST https://api.deepseek.com/v1/chat/completions
  │               ├─ 3次重试，指数退避 2^attempt 秒
  │               └─ 返回 LLMResponse(content, tool_calls, finish_reason,
  │                                    reasoning_content, usage)
  │
  ├─ 无工具调用时 ────────────────────────────────
  │ finish_reason=="length" → "请继续。" → continue
  │ finish_reason=="error"  → sleep 1.5s → continue
  │ 其他 → break (任务完成)
  │
  ├─ 工具执行 ────────────────────────────────────
  │ agent.py:331  for tc in response.tool_calls:
  │   tr = await _execute_tool(tc, session_id)
  │   └─ agent.py:1062  _execute_tool()
  │       ├─ ToolRegistry.get(tc.name)
  │       ├─ ToolGuardrails.check_tool_call(name, args, risk)
  │       │   ├─ 注入检测: 扫描 <system_reminder> 等 20+ 模式
  │       │   ├─ 危险命令: 50 种正则 (rm -rf /, fork bomb, ...)
  │       │   ├─ 路径沙箱: 阻止写入 /etc, C:\Windows
  │       │   └─ 按工具名/action 分派检查
  │       ├─ bash/shell → Sandbox.execute(command, mode="local")
  │       │   └─ asyncio.create_subprocess_shell
  │       ├─ 其他工具 → RetryManager.execute_with_retry(handler, **args)
  │       │   └─ 指数退避+jitter, 权限/不存在错误不重试
  │       ├─ Auditor.record_tool_call / record_file_access
  │       └─ ResultClassifier.classify(name, result, error, duration, timeout)
  │           ├─ "permission denied" → denied
  │           ├─ "timeout"/"429" → timeout/rate_limited
  │           ├─ "not found" → partial
  │           └─ 默认 → success / failure
  │
  ├─ 更新工作记忆 ────────────────────────────────
  │ _update_working_memory(response.content, round_results, user_message)
  │   └─ agent.py:957
  │       ├─ ExecutionPlan.parse_from_text() → JSON 计划解析
  │       ├─ 每个 ToolResult:
  │       │   ├─ wm.record_attempt()
  │       │   ├─ 失败 → error_kb.lookup() → wm.record_error()
  │       │   ├─ 成功+上次失败 → error_kb.learn() 知识库学习
  │       │   ├─ verifier.verify_tool_result()
  │       │   │   ├─ shell: stderr/exit_code/17种错误信号
  │       │   │   ├─ file: Path.exists()/空文件检测
  │       │   │   ├─ web: 空响应/超时
  │       │   │   └─ _analyze_shell_error(): 24种→修复建议
  │       │   ├─ wm.record_verification()
  │       │   └─ 代码文件 → wm.code_writes + 编译提示
  │       ├─ PlanStep.mark_done()
  │       └─ 连续失败 → wm.rule_out()
  │
  └─ 追加消息 ────────────────────────────────────
      assistant_msg = Message(ASSISTANT, content, tool_calls, reasoning_content)
      for tr: Message(TOOL, content=str(result)[:4000], tool_call_id=tc.id)
```

### 阶段 5：后处理

```
agent.py:372  report = _build_task_report(task, rounds, results, ...)
              → "✅ 成功 / ⚠️ 部分完成 / ❌ 失败 / 🛑 达到上限"

agent.py:381  sessions.add_message()  → 会话历史持久化

agent.py:386  memory.observe_action()         ← L4 预测记忆学习
agent.py:387  memory.store_conversation()     ← L3 FTS5 存储
agent.py:389  memory.remember(FROZEN)         ← L1 冻结快照

agent.py:392  if auto_generate: _learn_from_task()
              → 构建 trace → procedural.create_from_trace()
              → GEPAEngine.evolve()（如启用）

agent.py:396  if success: checkpoint_mgr.mark_complete(sid)
              else: _save_checkpoint(sid, task)

agent.py:402  error_kb.save()         ← 持久化错误知识库

agent.py:404  sessions.update_state(sid, IDLE)

agent.py:414  finally: lane_queue.mark_done(sid)  ← 释放 GlobalLane (实际 L416)
```

### 阶段 6：流式变体（chat_stream vs chat）

| 步骤 | chat() | chat_stream() |
|------|--------|---------------|
| agent.py:428 | — | `lane_queue.track_active(sid)` |
| 状态 | THINKING | STREAMING |
| Steer | — | `register_steer_callback(sid, cb)` |
| LLM调用 | `llm.chat()` → LLMResponse | `llm.chat_stream()` → AsyncIterator |
| 工具结果 | 不输出 | `yield StreamChunk(tool_result=tr)` |
| 释放 | `mark_done()` | `untrack_active()` + `mark_done()` |

---

## 二、文字层面：以 "你好" 为例的完整流转

### 用户输入

在终端输入 `deepsoul chat "你好"`。cli.py 第 74-87 行，检测到 message 参数非空，走单次消息模式。

### ① 入队——几乎没有等待

"你好" 被包装成 `QueueItem(prompt="你好", mode=ADAPTIVE)`。Lane Queue 检查当前状态：没有流式任务、没有积压、全局并发槽位有空余。直接分配槽位，耗时 < 50ms。

### ② 记忆检索——四层全空

Agent 调用 `memory.query_for_prompt("你好")`：

- **L2 技能库**：关键词匹配，"你好"不在任何技能的触发词里，返回空
- **L3 FTS5**：全文搜索历史对话，新会话数据库里没记录，返回空
- **L4 预测记忆**：第一天用，行为图是空的，返回空
- **用户模型**：还没学过偏好，返回空

四个层全部返回空字符串。`matched_skills` 也是空列表。

### ③ Prompt 组装——11 段 XML 拼接

`PromptBuilder.build_system_prompt(matched_skills=[], tools=工具声明, extra_context="")` 跑完 11 段：

SOUL.md → AGENTS.md → TOOLS.md → 工具声明 → 安全规则 → 200+ 行全局行为规则。全局规则里有一条："对话/问候直接文字回复，不要调工具"。

组出大约 3000 token 的 system prompt。首次调用触发快照冻结，后续调用读冻结内容保护 prefix cache。

### ④ 增强注入——首轮侦察指令

`_build_enhanced_prompt(base_prompt, "你好", first_round=True)`：

- 工作记忆空，跳过
- 检查点：`load_latest(max_age_hours=1)` 无结果，跳过续跑
- 侦察指令：注入 CWD，判断 `_is_vague_task("你好")` → 不是模糊词，走"摸底后动手"路径
- 编码节拍：不是编码任务，跳过
- JSON 计划：注入模板

然后把 `working_memory.clear()` 重置。

### ⑤ LLM 推理——0 个工具调用

请求体约 4000 token，POST 到 DeepSeek API。返回：

```json
{"choices":[{"message":{"content":"你好！有什么可以帮你的吗？"},"finish_reason":"stop"}]}
```

`response.tool_calls` 是空的，`finish_reason` 是 `"stop"`。第一轮就 break 出循环了。

### ⑥ 后处理——记忆写入

- 任务报告：`results` 是空的，`_build_task_report()` 返回 ""
- 会话持久化：user/assistant 两条消息写入 session
- 记忆：`observe_action("你好")` → L4 学到一个行为；`store_conversation()` → L3 存两条消息；`remember(摘要, FROZEN)` → L1 冻结快照
- 技能学习：trace 只有 0 步，不够 2 步门槛，跳过
- 检查点：`consecutive_fails=0 < 5`，`mark_complete()` 删除检查点
- 错误知识库：`save()` 持久化

### ⑦ 释放

`finally` 块：`lane_queue.mark_done(sid)` → `GlobalLane.release()` 释放信号量。

终端打印 "你好！有什么可以帮你的吗？"。

---

## 三、对比：复杂任务 vs 简单问候

| 阶段 | "你好" | "帮我写一个 FastAPI 项目" |
|------|--------|--------------------------|
| 记忆检索 | 全空 | 可能匹配到 python_dev/web 等技能 |
| 增强注入 | 跳过编码节拍/纠错 | 注入"小步快跑"+JSON 计划 |
| LLM 调用 | 1 轮 | 20-30 轮 |
| 工具调用 | 0 次 | 30-50 次 (file/bash/web) |
| 工作记忆 | 无更新 | 每轮更新尝试/错误/验证/计划进度 |
| 检查点 | 无 | 每轮保存，完成后删除 |
| 技能学习 | 跳过 | trace ≥ 2 → 自动生成 .skill |
| 总耗时 | ~3 秒 | 2-5 分钟 |

---

## 四、涉及的 25 个模块

```
cli.py → agent.py → lane_queue.py → session.py
                 → memory/manager.py → frozen/procedural/indexed/predictive/user_model/error_kb
                 → prompt/builder.py → compressor.py
                 → engine/working_memory.py → verifier.py → checkpoint.py
                 → llm/registry.py → deepseek.py → base.py
                 → tools/registry.py → guardrails.py → retry.py → classifier.py
                 → safety/sandbox.py → auditor.py
                 → types.py
```
