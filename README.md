# OUI-X

OUI-X 是基于 **Open WebUI** 的二次开发分支，目标是：

* **更清晰、更可控的 RAG / Knowledge Base 能力**
* **原生（native）工具调用流水线**（含 MCP）
* **OpenAI Responses API 适配**
* **更轻量、更稳定的默认构建与运行体验**（镜像从 4.7GB -> 2.6GB）

## 项目定位

* **更轻、更干净的默认发行版**：移除大量默认不使用或维护成本高的引擎/入口（Ollama、部分语音引擎、OpenAPI tools、评测/评分/Arena 等）。
* **工具调用优先**：把工具体系升级到以 **MCP（Model Context Protocol）** 为核心，支持系统级与用户级 MCP server 管理、OAuth 2.1、SSE/Streamable HTTP。
* **RAG 强化**：从“一个全局配置套所有集合”升级到“**每个 Knowledge Collection 可独立覆写配置**”，并强化 clone、BM25/Hybrid/Rerank 的配置结构与稳定性。
* **面向 Responses API**：新增 OpenAI Responses API 的适配层，支持对话、图片输入、工具调用与 reasoning/thinking 的展示。

## 新增了什么？

### 1) 原生 MCP（Native MCP Calls）

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

### 2) OpenAI Responses API 适配

OUI-X 新增 Responses API 的适配层（在后端与前端均有接入）：

* chat messages → responses input 的转换
* responses streaming event → chat SSE chunks 的映射
* tool calls / tool followups 在 Responses 结构里的表达与回放
* reasoning/thinking 的提取与展示（同时避免把不合适的 summary 混入 UI）
* UI 展示层对 content blocks 做了重排：**thinking 在正文之前展示**（仅展示顺序调整）

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

### 4) Tool Calling Timeout（工具调用超时）

新增 **工具调用超时**配置，并在后端 middleware 端到端注入：

* `TOOL_CALL_TIMEOUT_SECONDS` 作为全局持久化配置
* 用户 UI settings 可覆写（并做 clamp/兜底）
* tool calling 相关 metadata 同时包含：
  * `max_tool_calls_per_round`
  * `tool_call_timeout_seconds`

### 5) RAG / Knowledge Base 强化

#### 5.1 Retrieval Query Generation：参考上下文轮数（turns）

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

#### 5.3 Hybrid/BM25/Reranking 配置结构重构

* 将配置从“混合概念”拆得更清晰：
  * BM25 权重独立（例如从 `RAG_HYBRID_BM25_WEIGHT` 到 `RAG_BM25_WEIGHT`）
  * BM25 搜索、enriched texts、reranking 分别有开关
* 新增 **Voyage reranking engine**
  * engine 校验支持 `external` / `voyage`
  * external reranker 的 payload/response 结构更兼容

#### 5.4 对话框一键禁用 RAG（端到端阻断）

新增一个聊天输入区按钮：**Disable RAG**
* 前端：作为 feature 写入请求
* 后端：真正端到端 short-circuit：
  * 不走 model knowledge
  * 不触发 web_search
  * 不处理文件 sources
* 目标：让用户能稳定地把“工具/模型对话”与“RAG 对话”分离开

#### 5.5 Collection 级别的 RAG 独立配置覆写

OUI-X 的一个关键变化：**每个 Knowledge Collection 可以在自身 meta 里保存配置覆写**，从而让不同 collection 使用不同策略（embedding/rerank/阈值/top_k 等）。

实现层面：

* 后端检索改为“逐 collection 按 effective config 查询，再 merge 排序”
* UI 增加 collection config modal（用于编辑/保存覆写项）
* config 访问权限对齐：**需要 write 权限**（只读用户不再尝试加载 config，避免 toast 噪音）

#### 5.6 新增 Knowledge Collection Clone

新增 Knowledge clone API：

* `POST /knowledge/{id}/clone`
* 优先 direct clone（向量库复制），失败回退 re-embed
* 返回 `warnings`：提示 fallback 原因与部分失败细节
* 对 Chroma：
  * 复制策略升级为更可恢复的复制方式（按 ID copy、指数退避重试）
  * 部分失败文件 fallback re-embed
  * 修复 `has_collection` 兼容不同 Chroma client 返回形态

#### 5.7 新增 Knowledge Collection Download

新增 Knowledge download API：

* 优先读取文件实体关联的 storage 文件并写入 zip
* 若 storage 文件缺失/不可读，fallback 到向量库内容并生成 *.fallback.txt
* fallback 查询顺序：file-{file_id} -> 当前 knowledge active collection（按 file_id 过滤）
* 对单文件失败采用“不中断整体导出”策略，失败详情写入 manifest

#### 5.8 Chroma 引入 Redis 队列写入调度

* 新增 Chroma Redis 开关（默认关闭）：
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

#### 5.9 Conversation File Upload Embedding + 安全清理

* 新增 Conversation File Upload Embedding 开关链路（默认关闭）：
  * 新增全局开关 Conversation File Upload Embedding（默认关闭）
  * 新增用户级设置 Conversation File Upload Embedding（仅显式 true 时覆盖开启）
  * 当用户级设置关闭时，设置将跟随全局状态
* 会话文件上传路径策略变化：
  * 关闭 embedding 时：会话文件走“仅抽取文本 + full-context/direct_context”路径，不写入会话知识库
  * 开启 embedding 时：会话文件进入用户专属 conversation knowledge collection
  * 上传路径策略补充：Knowledge 上传走 vector_db/uploads；会话图片始终走 uploads；其余会话文件按开关决定

#### 5.10 删除清理链路增强

* 新增支持关联引用、文件实体、存储文件与数据库记录的级联清理，并补充确认交互弹窗
* 当文件仍被其他 Knowledge Collection 引用时，跳过文件实体删除，仅执行当前集合 detach，避免误删共享文件
  * warnings 中返回 skipped 信息，便于审计与排障

#### 5.11 Conversation Add To Collection（会话一键入库）

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

### 6) 高级参数侧边栏重构与 Responses 参数规范化

* 高级参数面板收敛与重排：
    * 新增并突出 reasoning_effort、verbosity、summary 三个参数入口（统一为标准化值）
    * Provider 生效条件：
        * 仅在 provider_type = openai_responses 时透传 summary/verbosity 等 Responses 语义参数
        * 非 Responses provider 会清理 summary 相关字段，避免不兼容参数污染请求
    * 移除旧采样参数在侧边栏与提交链路中的默认透传（如 min_p、repeat_penalty、tfs_z、mirostat*、use_mmap/use_mlock）
* 参数标准化规则（前后端一致）：
    * reasoning_effort：去空白并转小写；空值不下发
    * verbosity：去空白并转小写；空值或 none 不下发
    * summary：去空白并转小写；空值或 none 不下发
* 前端UI展示标准化为 Title Case，统一移除下划线
* 每个模型预配置中的 Advanced Params 优先级从最高级降为“仅 fallback”，用户请求参数始终优先

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

## 重构与稳定性改进（值得一提）

* Docker 构建与依赖体系多次“瘦身”：
  * 默认不启用/不安装一批未使用的依赖（减少镜像体积与构建失败）
* Streaming 资源回收修复：
  * Responses API 流式转发场景加入后台 cleanup，避免长跑线程/连接泄漏
* Chat UI quick actions（浮动按钮）修复：
  * 请求体更标准化、stream 解析更稳、错误信息更可诊断
  * 后端对 `parent_message` 做了类型兜底防崩

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
