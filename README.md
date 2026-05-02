# OUI-X

OUI-X 是基于 **Open WebUI** 的二次开发分支，目标是：

* **更清晰、更可控的 RAG 能力**
* **原生工具调用流水线**
* **OpenAI Responses API 适配**
* **更轻量、更稳定的默认构建与运行体验**（镜像从作为修改基础的 v0.6.43 ~4.7GB -> 2.29GB）

## 项目定位

* **更轻、更干净的默认发行版**：移除大量默认不使用或维护成本高的引擎/入口（Ollama、部分语音引擎、OpenAPI tools、评测/评分/Arena 等）。
* **工具调用优先**：把工具体系升级到以 **MCP** 为核心，支持系统级与用户级 MCP server 管理、OAuth 2.1、SSE/Streamable HTTP。
* **RAG 强化**：从“一个全局配置套所有集合”升级到“**每个 Knowledge Collection 可独立覆写配置**”，并强化 clone、BM25/Hybrid/Rerank 的配置结构与稳定性。
* **面向 Responses API**：新增 OpenAI Responses API 的适配层，支持对话、图片输入、工具调用与 reasoning/thinking 的展示。
* **连接与协议收敛**：连接 provider 已收敛为 `openai` / `openai_responses` 两类；Responses/Chat 采用并行原生链路，但路由决策与 payload 构建统一由后端适配层收敛；`Local Connection` 与 `Azure OpenAI provider` 均已下线为历史兼容语义。
* **前端性能优化**：做了一系列修改优化，减少因前端性能问题造成的卡顿。

## 新增了什么？

### 1) 新增原生 MCP（Native MCP Calls），同时移除原有 OpenAPI 体系

OUI-X 增加了“原生 MCP 工具调用”的完整链路：

* 支持 **Streamable HTTP** 与 **SSE** 两种 MCP 传输方式
* 支持 MCP server 的 **start/stop（按工具列表启停）**
* 支持 **OAuth 2.1**（动态注册 + 回调 + token 管理）
* MCP tool schema/返回内容做了“更 OpenAI 友好”的标准化处理（schema 归一化、content block 序列化兜底）
* 支持两种管理模式：
  * **系统级 MCP 连接**（管理员配置）
  * **用户级 MCP 连接**（用户可在 UI 中添加、verify、完成 OAuth 授权）
* 在工具调用结果标题行左侧新增 Disable Context Injection 开关
  * 开关仅在工具调用完成时展示，点击开关不会触发展开/折叠行为
  * 开关默认关闭，打开时将禁用本条工具调用数据注入上下文行为
  * 每个开关状态随会话数据在后端持久化保存

### 2) 新增原生 OpenAI Responses API 适配

OUI-X 的 Completion/Responses 已收敛到统一适配层（`completion_adapter`），并由 `provider_type` 驱动端点分流：

* 端点路由矩阵（主聊天链路）：
  * `openai_responses -> /api/responses`
  * `openai -> /api/chat/completions`
* 协议事实源：
  * 前端 `endpointKind` 仅作为 hint，最终以后端适配层决策为准
  * 流式/非流式共享同一套端点决策与请求构造逻辑，减少分支漂移
* 前端请求构造原生化：
  * Responses 请求使用原生 schema（`input/tools/tool_choice/reasoning/verbosity/metadata` 等）
  * Chat 请求维持 `chat/completions` schema
  * 显式使用 `endpointKind` + 联合类型，避免隐式猜测路由
* 统一适配能力：
  * 统一端点决策（`chat_completions` / `responses`）
  * 统一上游 payload 构建
  * 统一 messages -> responses.input 转换
  * 统一 tools 结构归一化（兼容 chat-style / responses-style）
  * 统一 prompt cache 策略注入
* 后端 `/api/responses` 接入统一聊天任务流水线：
  * 复用 `process_chat_payload / process_chat_response / create_task`
  * 保持 `task_id`、socket 事件、标题/标签/follow-up 等行为一致
* 端点与 provider 契约：
  * `/responses` 仅允许 `openai_responses`
  * `/chat/completions` 面向 `openai`
* 透传策略统一：
  * 请求与响应未知字段默认不裁剪
  * 仅剥离内部控制字段与非上游协议字段
* 多入口一致性（统一适配层贯通）：
  * 主对话入口
  * middleware 工具 follow-up 链路
  * 任务链路（title/tags/follow_up/query/autocomplete/emoji/moa）
  * 前端流式 quick actions 调用
  * Socket 直连路径 `request:chat:completion / request:responses:completion` 按 provider 执行同一分流规则
* 围绕 Workspace Models 的自定义模型链路做了系统性重构：
  * 自定义模型运行时继承基础模型 provider_type，避免仅凭 custom model id 误判 provider
  * 三层参数合并优先级：请求级 > 自定义模型参数 > 基础模型参数；空值不参与覆盖；合并统一发生在最终payload构建前
  * 当 custom model 对应 base model 为 openai_responses 时，后端自动切到 responses 链路
  * 模型 system 映射到 instructions（而非注入 messages）；response_format 兼容映射到 text.format
  * 支持 base_model_id 递归解析真实 provider，避免 tasks 误走 chat 链路
* 流式消费：
  * 同时支持 Chat SSE 与 Responses SSE
  * 统一输出文本/工具调用/reasoning/usage 事件
  * 保留 raw event 通道，未知字段不丢失
  * responses streaming event -> chat SSE chunks 的映射
* tool calls / tool followups 在 Responses 结构里的表达与回放
* Responses API 的 reasoning summary 在流式/非流式场景下的提取与展示时序：
  * 后端 middleware 新增 reasoning summary 流状态映射，支持 response.reasoning_summary_* 事件增量返回
  * 前端 SSE 解析新增对应事件分支，实时 yield reasoning
  * multi-part summary 增加分段换行衔接，提升可读性
  * reasoning 展示仅采信 `output[].summary[].text`，不读取 `response.reasoning.summary` 这类请求参数回显字段
  * 流式已产出时 completed 不重复注入；未流式产出时从 output[].summary 回填
  * UI 展示层对 content blocks 做了重排：**thinking 在正文之前展示**（仅展示顺序调整）
* 缓存命中功能增强，大幅提升缓存命中率：
  * 请求未显式传 `prompt_cache_key` 时，服务端自动注入会话级 key，无需前端/用户手动传
  * 新会话自动生成 key，同会话稳定复用，跨会话隔离
  * 显式透传 `prompt_cache_key` 优先，不会被自动注入逻辑覆盖（便于灰度/调试）
  * 临时会话使用稳定派生 key，不写入数据库，持久会话将 key 写入会话元数据

### 3) Native-Only 工具调用流水线重构

围绕“只走 native function calling”的 pipeline 做了较大重构与测试补齐：

* 新增并发控制：
  * `TOOL_CALL_MAX_CONCURRENCY`（默认值与健壮兜底）
* 新增全局/用户级工具调用配置注入（metadata 中可区分 global/user）
* 关键工具编排逻辑补齐单测：
  * function schema 规范化
  * 参数解析 fallback
  * 并发峰值验证
  * follow-up messages 结构验证

### 4) 新增 Tool Calling Timeout 控制

新增 **工具调用超时**配置，并在后端 middleware 端到端注入：

* `TOOL_CALL_TIMEOUT_SECONDS` 作为全局持久化配置
* 用户 UI settings 可覆写（并做 clamp/兜底）
* tool calling 相关 metadata 同时包含：
  * `max_tool_calls_per_round`
  * `tool_call_timeout_seconds`

### 5) RAG 全链路重构及强化

#### 5.1 新增 Retrieval Query Generation：参考上下文轮数（turns）

新增配置项：生成检索 query 时参考多少轮历史对话：

* `RETRIEVAL_QUERY_GENERATION_REFER_CONTEXT_TURNS`
* 明确 **排除最新一条 user query**，只取前面的上下文
* 附带单测覆盖（turns=0、排除最新 user 等）

#### 5.2 Chunking / Splitter：引入 Voyage tokenizer + warmup

* 新增 `VOYAGE_TOKENIZER_MODEL` 与更新逻辑
* `token_voyage` splitter 场景下：
  * 没配模型会回填默认值
  * splitter 变化时异步 warmup tokenizer，降低首次请求抖动
* 增强 tiktoken encoder / voyage tokenizer 的缓存与预热能力

#### 5.3 重构拆分 Hybrid / BM25 / Reranking

* 将配置从“混合概念”拆得更清晰：
  * BM25 权重独立（例如从 `RAG_HYBRID_BM25_WEIGHT` 到 `RAG_BM25_WEIGHT`）
  * BM25 搜索、enriched texts、reranking 重构拆分，分别开关控制
* 新增 **Voyage reranking engine**
  * engine 校验支持 `external` / `voyage`
  * external reranker 的 payload/response 结构更兼容

#### 5.4 新增对话框一键禁用 RAG（端到端阻断）

新增一个聊天输入区按钮：**Disable RAG**
* 前端：作为 feature 写入请求
* 后端：真正端到端 short-circuit：
  * 不走 model knowledge
  * 不触发 web_search
  * 不处理文件 sources
* 目标：让用户能稳定地把“工具/模型对话”与“RAG 对话”分离开

#### 5.5 新增 Collection 级的 RAG 独立配置覆写

OUI-X 的一个关键变化：**每个 Knowledge Collection 可以在自身 meta 里保存配置覆写**，从而让不同 collection 使用不同策略。
Collection Config 按 **Embedding / Retrieval / BM25 / Reranking / Other** 等大类别分组。

* **支持的 Collection 级覆写参数**
  * **Embedding Config：**
    * Embedding Model Engine：嵌入引擎
    * Embedding Model：嵌入模型
    * Text Splitter：分词器
    * Voyage Tokenizer Model：指定 Voyage 模型专用分词器
    * Chunk Size：块大小
    * Chunk Overlap：块重叠大小
    * Embedding Batch Size：批大小
  * **Retrieval Config：**
    * Top K：向量直出召回候选数量
    * Retrieval Chunk Expansion：向前向后扩展 Chunk 数量
  * **BM25 Config：**
    * BM25 Search：开启 BM25 混合检索
    * Enrich BM25 Text：开启增强检索
    * BM25 Weight：BM25 权重
  * **Reranking Config：**
    * Reranking：开启重排
    * Reranking Engine：重排引擎
    * Reranking Model：重排模型
    * Top K Reranker：重排返回数量
    * Relevance Threshold：最低置信度阈值
  * **Other Config**
    * RAG Template：RAG 回复模板提示词

实现层面：

* 后端检索改为“逐 collection 按 effective config 查询，再 merge 排序”
  * `k` 参数解析优先级统一为：**显式 `k` > collection override 的 `TOP_K` > 全局 `TOP_K`**
  * 多 collection 且未显式传 `k` 时，merge 上限按 `max(collection_k)` 解析
* BM25 Search 和 Reranking 开关的生效边界：
  * BM25 Search 关闭时，后端会忽略 BM25 enriched texts/BM25 weight 对检索路径的影响
  * Reranking 关闭时，后端会忽略 reranking engine/model、top_k_reranker、relevance_threshold
* config 访问权限对齐：**需要 write 权限**（只读用户不尝试加载 config，避免 toast 噪音）
* 默认内置 `RAG Template` 升级为 retrieval-grounded 风格：
  * 强调“上下文优先”，并要求在上下文缺失时显式说明缺失边界
  * 上下文冲突时要求并列呈现冲突点并标注不确定性
  * 更改 citation 输出规范为 `[[id]]` / `[[id1,id2,id3]]`

#### 5.6 新增 Knowledge Collection Clone 能力

新增 Knowledge clone API：

* `POST /knowledge/{id}/clone`
* 优先 direct clone（向量库复制），失败回退 re-embed
* 返回 `warnings`：提示 fallback 原因与部分失败细节
* 对 Chroma：
  * 复制策略升级为更可恢复的复制方式（按 ID copy、指数退避重试）
  * 部分失败文件 fallback re-embed
  * 修复 `has_collection` 兼容不同 Chroma client 返回形态

#### 5.7 新增 Knowledge Collection Download 能力

新增 Knowledge download API：

* 优先读取文件实体关联的 storage 文件并写入 zip
* 若 storage 文件缺失/不可读，fallback 到向量库内容并生成 *.fallback.txt
* fallback 查询顺序：file-{file_id} -> 当前 knowledge active collection（按 file_id 过滤）
* 对单文件失败采用“不中断整体导出”策略，失败详情写入 manifest

#### 5.8 新增 Chroma Redis 队列写入调度

* 引入 Chroma Redis，新增 Chroma Redis 变量开关（默认关闭）：
  * `CHROMA_REDIS_ENABLED`（仅显式 true 时启用）
  * `CHROMA_REDIS_URL`（默认回退 `REDIS_URL`）
* 未启用时保持原行为不变，确保向后兼容
* VECTOR_DB="chroma" 且 CHROMA_REDIS_ENABLED="true" 且 CHROMA_REDIS_URL 非空时，才会生效
* 增强队列超时可观测性：
  * 超时日志补充 in_event_loop、thread 与队列/leader/pending 快照
  * 更易定位“消息未消费 / leader 丢失 / 事件循环阻塞”问题
* 将 async 路由内同步向量写操作线程池化（run_in_threadpool）：
  * 覆盖 knowledge/files/memories 的 upsert/delete/delete_collection/reset 等路径
  * 降低 Clone/Delete 场景下请求超时与卡死风险
  * direct clone（qdrant/chroma）使用线程池执行，降低大批量复制对事件循环的阻塞

#### 5.9 Conversation File Upload 链路拆分为 Upload / Embedding

* 新增 Conversation File Upload Embedding 开关链路（默认关闭）：
  * 新增全局开关 Conversation File Upload Embedding（默认关闭）
  * 新增用户级设置 Conversation File Upload Embedding（仅显式 true 时覆盖开启）
  * 当用户级设置关闭时，设置将跟随全局状态
* 会话文件上传路径策略变化：
  * 关闭 embedding 时：会话文件走“仅抽取文本 + full-context/direct_context”路径，不写入会话知识库
  * direct_context 文件只读取已提取的 `file.data.content`，不调用向量检索，不进入 RAG sources，也不会触发 `RAG_TEMPLATE`
    * direct_context 文件内容会作为普通 user message 上下文前置注入，注入顺序严格跟随当前请求中的 `metadata.files/files` 顺序
    * 用户从前端移除文件后，后续请求不再携带该文件，因此不会继续注入
    * 同一轮同时包含 direct_context 文件与 Knowledge/RAG sources 时，两条链路保持独立：上传文件走普通上下文，知识库/检索结果继续走 `RAG_TEMPLATE`
  * 开启 embedding 时：会话文件进入用户专属 conversation knowledge collection（自动创建）
  * 上传路径策略补充：Knowledge 上传走 vector_db/uploads；会话图片始终走 uploads；其余会话文件按开关决定

#### 5.10 重构安全删除清理链路

* 新增支持关联引用、文件实体、存储文件与数据库记录的级联清理，并补充确认交互弹窗
* 当文件仍被其他 Knowledge Collection 引用时，跳过文件实体删除，仅执行当前集合 detach，避免误删共享文件
  * warnings 中返回 skipped 信息，便于审计与排障

#### 5.11 新增 Conversation Add To Collection（会话一键入库）

* 在会话菜单新增 `Add To Collection` 按钮（Navbar / Sidebar 均支持）
  * 在 Add To Collection 弹窗中新增两个内容开关：
    * Add Thinking Content（默认开启）
    * Add Tool Calling Content（默认开启）
* 仅在满足以下权限时展示入口：
  * `workspace.knowledge` 可用（或 admin）
  * `chat.file_upload` 可用（或 admin）
* 弹窗仅展示有写权限（`write_access`）的 Knowledge collections，并支持分页加载
* 选定 collection 后将当前会话序列化为 Markdown，按标准上传链路写入知识库：
  * 若“上传成功但挂载到知识库失败”，会自动执行文件清理（`deleteFileById`）以避免脏数据残留
  * 当无可写 collection 时，提示并引导跳转到 Knowledge 工作区创建/管理
* Add To Collection 文件命名规则，导出文件名由：
  * `chat-<yyyyMMdd-HHmm>-<会话名>.md`
  * 文件名安全清理逻辑：
    * 替换非法字符（如 `/ \\ : * ? " < > |` 及控制字符）为 `-`
    * 去除首尾空白与尾部 `.`，避免跨平台文件名兼容问题
  * 当标题为空或清理后为空时，统一回退为 `chat`
  * 后端存储策略：由后端追加 `uuid_` 前缀（如 `uuid_chat-<yyyyMMdd-HHmm>-<原会话名>.md`）

#### 5.12 重构 RAG 召回链路

对 Retrieval 链路做了“行为一致性”收敛，覆盖非 rerank 场景与多 collection 合并语义：

* 融合召回统一在最后一步进行一次去重
  * 去重策略保序，优先使用 metadata 键；缺失时回退归一化文本哈希
  * `retrieval -> dedupe -> optional rerank -> expansion -> dedupe -> output`
* merge 上限增加 expansion-aware 解析：
  * 未显式传 `k` 时，按 `TOP_K * (2 * expansion + 1)` 计算每集合上限并取 `max`
  * 显式传 `k` 时保持原语义，避免破坏调用行为

#### 5.13 新增 Retrieval Chunk Expansion

* 新增 Retrieval Chunk Expansion 全局配置：
  * 增加 `RETRIEVAL_CHUNK_EXPANSION`（`rag.retrieval_chunk_expansion`，默认 `0`）
  * 配置值更新时增加 `0~100` clamp 保护
* 调整 Chunk Expansion 执行时机为统一后置阶段：
  * 检索链路调整为 `retrieval -> dedupe -> optional rerank -> expansion -> dedupe -> output`
  * Chunk Expansion 不再依赖 Vector / BM25 / Rerank 开关，只要 expansion > 0 即执行
  * 主链路与 file-scope 链路保持一致，避免扩展结果放大 rerank 输入规模
* 优化扩展结果的可控性与可观测性：
  * 扩展后增加结果上限控制：`final_limit = max(k, k * (2N + 1))`
  * 日志统计项补充为：
    * `retrieved_count`
    * `deduped_count`
    * `rerank_input_count`
    * `rerank_output_count`
    * `expanded_output_count`
* 对齐 UI 交互体验：
  * 为 Retrieval Chunk Expansion 增加并完善 Tooltip 提示文案
  * Tooltip 交互结构与 BM25 Weight 对齐，支持整行 hover 与 inline-tooltip

#### 5.14 Citation 体验升级与标签语义统一

Citation 行为重构为“正文可精确点击 + 底部聚合不变 + 标签语义一致”：

* 引用编号从 source 粒度升级为 chunk 粒度：
  * 正文 `[[n]]` / `【【n】】` 可精确定位到对应 chunk
  * 底部 Sources 仍按 source 聚合展示
* 正文引用按钮文案升级为“源文件名[n]”：
  * 多引用主按钮展示 `源文件名[n]+k`
  * 保留 source 聚合面板交互，降低阅读负担
* 引用标签支持“可点击但不可选中/复制”，减少正文复制时的噪音
* 引用语法兼容半角/全角及混合相邻形式（如 `[[1]][[2]]`、`【【1】】【【2】】`）
* 单层括号内容（如 `[1]`、`【1】`、`[^1]`）不作为 citation token 解析，避免误判/误删
* Citation 弹窗标签语义统一展示为：**Extended > Retrieval > Relevance**
  * 无 rerank 场景下，原始召回块稳定显示 `Retrieval`
  * 扩展块稳定显示为 `Extended`，避免出现无标签或数值误导
  * rerank 分数场景下，展示百分比分数

#### 5.15 复制/导出净化链路统一

围绕消息复制与导出链路，新增统一净化模块并默认启用 citation 剔除能力：

* 复制按钮、自动复制、快捷键复制、TXT 导出、PDF 导出、Add To Collection 统一复用导出净化逻辑
* 导出默认按 source 范围剔除 citation token，避免把引用按钮文本混入正文
* 保留 footnote 与普通方括号文本，降低误伤风险（例如 `[^1]`、普通 `[2026]` 文本）
* Add To Collection 上传的 Markdown 默认不包含 citation token

#### 5.16 重构聊天 PDF 导出链路（Markdown -> HTML -> Print）

PDF 导出从旧方案重构为“Markdown 渲染打印”：

* 统一流程：`Markdown -> HTML -> 浏览器打印`
* 基于现有 marked 扩展渲染（details/katex/citation/footnote/mention/strikethrough）
* 导出内容与 markdown 文本链路对齐，文本可选中，多页排版更稳定
* 下载菜单文案调整为 `Print as PDF (.pdf)`

### 6) 重构高级参数侧边栏 / 模型参数配置项

* 高级参数面板收敛与重排：
    * 新增并突出 reasoning_effort、verbosity、summary 三个参数入口（统一为标准化值）
    * reasoning_effort 档位扩展为 7 档：`default` / `none` / `minimal` / `low` / `medium` / `high` / `xhigh`
    * 透传 summary/verbosity 等 Responses 语义参数
    * 非 Responses provider 会清理 summary 相关字段，避免不兼容参数污染请求
    * 移除旧采样参数在侧边栏与提交链路中的默认透传（如 min_p、repeat_penalty、tfs_z、mirostat*、use_mmap/use_mlock）
* 参数标准化规则（前后端一致）：
    * reasoning_effort：去空白并转小写；空值不下发
    * verbosity：去空白并转小写；空值或 none 不下发
    * summary：去空白并转小写；空值或 none 不下发
* 前端UI展示标准化为 Title Case，统一移除下划线
* 每个模型预配置中的 Advanced Params 优先级从最高级降为“仅 fallback”，用户请求参数始终优先
* Chat Controls 侧栏 `System Prompt` 交互增强：
    * `System Prompt` 输入区支持与对话输入框一致的 `/` Prompt 快捷建议，选中建议后会将当前命令位替换为对应内容，建议菜单在侧栏场景优先向下弹出，空间不足时自动翻转

### 7) 对话框新增 Reasoning Effort 滑块按钮

新增一个聊天输入区按钮：**Reasoning Effort 滑块**
* 点击灯泡按钮弹出轻量滑块，不影响原有 Tools / Integrations / Disable RAG 交互
* 档位与侧栏保持一致并双向同步：`default` / `none` / `minimal` / `low` / `medium` / `high` / `xhigh`
  * `default` 表示不下发 chat 级 `reasoning_effort`，回退使用模型预设 / fallback 参数
* 该控件直接读写当前 chat 的 `params.reasoning_effort`，发送链路不变（仍走现有 params 合并与下发）
* 拖动时显示当前档位提示，非拖动状态不显示提示；支持点击外部关闭与键盘操作（Enter/Space/Arrow）
* 滑块视觉采用“节点 + 已通过区段连线”样式

### 8) Chat 背景与代码块视觉增强

围绕聊天界面的背景可控性与代码块细节样式，新增以下能力：

* 背景上传能力增强：
  * 支持 `image/svg+xml` 类型作为聊天背景上传
  * 当浏览器未提供 MIME（`type === ''`）时，新增基于 `.svg` 后缀的兜底识别
* 双透明度控制（UI）：
  * 新增 `Chat Background Opacity` 与 `Chat Background Overlay` 两个控件
  * 两个控件均支持减号 / 滑杆 / 加号操作，步进为 `0.01`
* 渲染生效范围：
  * 背景图层透明度与背景蒙层透明度相互解耦，可分别调节
  * 仅作用于全局背景分支（用户设置或 license 背景），不影响 folder 专属背景分支
* CodeBlock 样式微调：
  * 对内层代码块容器圆角进行微调，外层容器保持不变

### 9) 重构文件/图片上传、转码与压缩链路

#### 9.1 图片压缩迁移到后端统一转码

原前端 canvas 方案低效缓慢且收益低，围绕聊天 / 频道 / 笔记中的图片上传链路，图片压缩从前端 canvas 缩放改为后端统一处理：

* `Settings -> File -> Image Compression` 保留，但压缩执行位置迁移到后端
* 用户关闭图片压缩时：图片原图直通，不做前后端额外转码
* 用户开启图片压缩时：
  * 图片先原始上传到后端
  * 后端按用户设置的宽高上限等比处理
  * 仅当原图宽或高超过限制时才缩放
  * 未超过限制时保持原始分辨率，但仍统一转为 `image/webp`
  * 输出质量新增用户设置项 `Image Compression Quality`（默认 `0.75`）
  * `compression_level` 固定为 `6`
* 当输入路径与最终 WebP 输出路径相同时，后端先转码到同目录临时文件，转码成功后再覆盖/更新为最终 WebP 文件，避免提前删除原文件导致 normalize orientation 阶段找不到输入文件
* 转码成功后只保留后端最终产物；会话展示图、文件内容接口返回图、远程 LLM 实际使用图保持一致
* 转码失败会自动重试 2 次，仍失败则标记文件失败并停止后续处理
* 新增按用户维度的 ffmpeg 并发限制，避免单个用户批量上传抢占全部转码资源
* 后端新增 HEIC / HEIF 解码支持，用于兼容苹果设备原图上传

#### 9.2 图片 base64 Data URL MIME

* 生成图片 Data URL 时，优先使用文件记录中的 `meta.content_type`
* 若 `meta.content_type` 缺失，再回退到基于文件名的 MIME 推断
* 若仍无法推断，则对图片回退到安全默认值 `image/webp`

#### 9.3 聊天附件上传增强

* 抽出聊天附件上传共享逻辑，主输入框与历史用户消息编辑态复用同一套处理流程
* 新增历史用户消息编辑态粘贴图片、粘贴文件、拖拽图片、拖拽文件支持
  * 拖拽到历史用户消息编辑框将会上传到历史用户消息编辑态中
  * 拖拽到其他空白处将会上传到底部新消息编辑态中
  * 编辑历史消息时，附件会写入当前编辑消息的 `editedFiles`，不会进入底部新消息输入框
  * 编辑态附件上传复用原有权限、模型能力、文件大小/数量、图片压缩、临时聊天等校验逻辑
  * 编辑态在附件仍处于上传中时阻止保存或重新发送，避免提交半完成状态
* 历史消息中的已上传图片预览统一通过 `content_type` 识别，并使用 `/files/{id}/content` 读取内容

#### 9.4 新增长文本粘贴自动转文件阈值配置

* `Settings -> Interface -> Paste Large Text as File` 开启后，会显示 `Large Text Character Limit`
* 默认阈值从 `1000` 调整为 `3000` 字符
* 阈值控件由滑杆 + 数字输入组成：
  * 滑杆范围：`100` 到 `999999`，步进 `1`
  * 数字输入允许任意正整数，不设最大值
  * 空值、`0`、负数、小数或非数字会回退到上一个有效值，默认回退 `3000`
* 粘贴纯文本长度严格大于阈值时才会转为 `.txt` 文件；等于阈值时仍保留为普通粘贴文本
* 聊天输入框内按住 `Shift` 粘贴时，仍会临时绕过“粘贴大文本为文件”逻辑

### 10) 前端性能整体优化改进

#### 10.1 新增消息列表虚拟化

* 重构 `Messages.svelte` 列表渲染，引入可见窗口计算、动态高度测量、占位区间渲染
* 新增虚拟窗口纯函数模块 `virtualization.ts`：可见范围索引计算、scrollTop 归一化、spacer 高度推导、边界不变量约束
* 滚动归一化保护与回写 guard，避免程序化回写导致滚动事件回环
* 虚拟化缓冲策略调整为稳态配置：`MESSAGE_OVERSCAN_PX = 2000`、`MESSAGE_OVERSCAN_ITEMS = 4`，结合视口动态放大 overscan 降低"卡边"风险
* 顶部加载更多从可见触发改为滚动阈值触发，减少虚拟化下误触发与抖动
* 空窗口兜底逻辑：边界场景至少渲染一条消息，避免整屏空白

#### 10.2 Markdown 渲染链路优化

* 新增聊天 Markdown 单例解析器 `lexChatMarkdown()`，`Marked` 实例集中注册扩展，避免每组件实例重复 `marked.use(...)` 初始化开销
* 流式解析节流（`STREAM_PARSE_THROTTLE_MS = 100`）：`done === false` 时按节流调度，减少 token 高频到达时的全量 lexer 次数；`done === true` 时立即完整解析
* 消除 `details` 分支模板期二次 lexer：`detailsTokenizer` 直接生成并挂载 `token.tokens`，删除 `marked.lexer(decode(token.text))` 路径
* `AlertRenderer` 引入 `WeakMap<Token, AlertData>` 缓存，同一 token 重渲染时避免重复 lexer

#### 10.3 流式渲染与更新调度优化

* Chat 流式链路改为批量刷新：新增流式消息更新队列与定时批量 flush（约 33ms），避免每个 chunk 直接写 `history` 触发渲染
* `chatEventHandler` 热路径移除不必要的 `tick`，状态/增量/引用等事件统一走调度更新
* `mergeResponses` 流式拼接路径改为批量更新，减少 token 级 UI 写入频率
* `scrollToBottom` 改为帧级合并调度：`requestAnimationFrame` 合并滚动写入，同一帧多次调用只执行一次；支持 smooth 行为优先级聚合

#### 10.4 TTS 与震动优化

* TTS 分句从"全量重算"改为"增量缓冲 + 节流"：每条消息新增 `deltaBuffer/sentenceCarry` 流状态，每次 chunk 仅追加增量，按节流窗口（约 180ms）做分句 flush
* 震动反馈增加节流（约 80ms），避免 chunk 级连续振动造成额外开销

#### 10.5 监听器与事件处理优化

* ContentRenderer 全局监听器按需注册：document 级 `mouseup/keydown` 从"组件挂载即注册"改为"浮动按钮显示时注册、关闭时解绑"
* `floatingButtons` 开关变化时同步绑定/解绑，避免多实例长期挂载全局处理函数

#### 10.6 缓存与短路优化

* `tool_calls` 解析增加快速短路：不含 `<details>` 或不含 `type="tool_calls"` 时直接返回
* 新增"上次内容 → 上次解析结果"缓存，内容不变时复用，避免重复正则扫描
* `update*` 系列函数增加快速短路，减少无意义 `replace`
* ResponseMessage 在切换注入状态时复用已计算的 `toolCallDetails`

#### 10.7 深比较热点替换优化

* `ResponseMessage`/`UserMessage`/`MultiResponseMessages` 移除高频 `JSON.stringify/parse` 深比较与深拷贝，改为直接引用 `history.messages[messageId]`
* `CodeBlock` 去掉 token 的 `JSON.stringify` 深比较，改为关键字段轻量比较
* `StatusHistory` 去掉 `JSON.stringify(statusHistory)` 比较，改为引用级更新策略

#### 10.8 模板计算响应式预计算优化

* `sourceIds/sourceLabels` 从模板 `reduce` 迁移到 `buildSourceMetadata` + 响应式变量
* `FloatingButtons` 的 `createMessagesList(history, messageId)` 改为响应式预计算 `floatingButtonMessages`
* `Message` 模板内重复 `history` 遍历与 `parent/siblings` 计算迁移到响应式预计算

#### 10.9 Collapsible 渲染期优化

* 移除组件实例内 `dayjs.extend` 和逐次 `locale` 加载逻辑，插件初始化上移至共享模块
* 工具调用数据的 `decode/parse/format` 从模板常量改为脚本缓存，仅在 `attributes` 实际变化时重算

#### 10.10 停止/取消/销毁阶段清理优化

* `stopResponse`、`chat:tasks:cancel` 与组件 `onDestroy` 时，清理 pending stream 队列与 TTS 定时器状态，避免遗留异步回调继续写入 UI

## 移除了什么？

### 1) 移除 Ollama 相关组件与“内置绑定”

* 去掉 Ollama 相关路由、前端管理组件与脚本
* README 不再宣传带 Ollama 的镜像/一键运行方式
* Docker build 不再在构建阶段安装 Ollama

### 2) 移除评测 / 评分 / Arena

* 删除 Evaluations 相关前后端页面与 API
* 删除 message rating 开关与 UI
* 清理相关权限点与配置项

### 3) 移除 OpenAPI tools（全面转向 MCP）

* 移除 OpenAPI tool servers 的 configs API、verify 流程
* 移除工具聚合里对 OpenAPI tool server 的注入
* middleware 不再根据 function_calling 决定注入 OpenAPI 工具

### 4) 语音能力收敛（移除部分 TTS / STT 引擎）

* TTS：移除 ElevenLabs 与部分本地 transformers TTS 分支
* STT：移除部分 provider/config（例如 Whisper/Deepgram/Mistral STT 等相关配置与依赖链）

### 5) 移除 Local Connection Type

* `Settings -> Connections -> Add/Edit Connection` 不再提供 Local/External 切换入口
* 新增/编辑连接固定写入 `connection_type=external`
* 历史配置中的 `connection_type=local` 在读取与聚合时自动归一化为 `external`
* 模型筛选与管理展示不再暴露 `(Local)` 语义

### 6) 移除 Azure OpenAI（Connections + RAG/Knowledge）

* Add/Edit Connection 不再提供 `Azure OpenAI` provider
* RAG Embedding Engine 不再提供 `azure_openai`
* Knowledge collection overrides 不再接受 `azure_openai`
* 历史 `azure_openai` 配置运行时自动归一化为 `openai`

### 7) 移除 Azure OpenAI 相关组件与链路

* 去掉 Provider Azure OpenAI 相关路由、前端管理组件

## 接口与兼容性变更（升级须知）

* Connections：
  * `OPENAI_API_CONFIGS[].provider_type` 不再支持 `azure_openai`
  * `azure` / `api_version` / `microsoft_entra_id` 等 Azure 遗留键不再生效，并在保存时清理
* Retrieval：
  * `GET /retrieval/embedding` 不再返回 `azure_openai_config`
  * `POST /retrieval/embedding/update` 不再接收 `azure_openai_config`
* 兼容策略：
  * 采用运行时归一化，无需一次性迁移脚本
  * 归一化仅针对本能力相关字段，不影响其他 Azure 功能（如存储/语音）

## 运行与开发（简述）

### 本地开发

* 前端：`src/`（Svelte）
* 后端：`backend/open_webui/`

你通常需要：

1. 启动后端（带必要环境变量）
2. 启动前端（指向后端 API）
3. 需要工具：配置 MCP 连接（系统级或用户级）

或者通过 Docker compose 进行构建测试

### Docker

OUI-X 的 Dockerfile 倾向于“更轻依赖、少副作用”。
如需特定能力（例如某些 OCR / STT / embedding 依赖），建议自行修改。

## 适用场景建议

* 想要把“工具生态”做成产品核心：优先使用 MCP + native pipeline
* 需要不同知识库集合使用不同检索策略：启用 collection-level config override
* 需要更现代的 OpenAI API 兼容：使用 Responses API 通路
* 希望默认构建更稳定、更轻：使用当前 OUI-X 的默认依赖策略
