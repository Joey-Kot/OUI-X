# OUI-X

OUI-X 是基于 **Open WebUI** 的二次开发分支，目标是：

* **更清晰、更可控的 RAG / Knowledge Base 能力**
* **原生（native）工具调用流水线**（含 MCP）
* **OpenAI Responses API 适配**
* **更轻量、更稳定的默认构建与运行体验**（镜像从 4.7GB -> 2.3GB）

---

## 项目定位

* **更轻、更干净的默认发行版**：移除大量默认不使用或维护成本高的引擎/入口（Ollama、部分语音引擎、OpenAPI tools、评测/评分/Arena 等）。
* **工具调用优先**：把工具体系升级到以 **MCP（Model Context Protocol）** 为核心，支持系统级与用户级 MCP server 管理、OAuth 2.1、SSE/Streamable HTTP。
* **RAG 强化**：从“一个全局配置套所有集合”升级到“**每个 Knowledge Collection 可独立覆写配置**”，并强化 clone、BM25/Hybrid/Rerank 的配置结构与稳定性。
* **面向 Responses API**：新增 OpenAI Responses API 的适配层，支持对话、图片输入、工具调用与 reasoning/thinking 的展示。

---

## 新增了什么？

### 1) 原生 MCP（Native MCP Calls）

OUI-X 增加了“原生 MCP 工具调用”的完整链路：

* 支持 **Streamable HTTP** 与 **SSE** 两种 MCP 传输方式
* 支持 MCP server 的 **start/stop（按工具列表启停）**
* 支持 **OAuth 2.1**（动态注册 + 回调 + token 管理）
* MCP tool schema/返回内容做了“更 OpenAI 友好”的标准化处理（schema 归一化、content block 序列化兜底）
* MCP tool 从对话框底部二级菜单重新移动到一级菜单，并添加计数提示
* 支持两种管理模式：

  * **系统级 MCP 连接**（管理员配置）
  * **用户级 MCP 连接**（用户可在 UI 中添加、verify、完成 OAuth 授权、自定义 Header）

---

### 2) OpenAI Responses API 适配

OUI-X 新增 Responses API 的适配层（在后端与前端均有接入）：

* chat messages → responses input 的转换
* responses streaming event → chat SSE chunks 的映射
* tool calls / tool followups 在 Responses 结构里的表达与回放
* reasoning/thinking 的提取与展示（同时避免把不合适的 summary 混入 UI）
* UI 展示层对 content blocks 做了重排：**thinking 在正文之前展示**（仅展示顺序调整）
* Responses API usage 返回信息补全兼容

---

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

---

### 4) Tool Calling Timeout（工具调用超时）

新增 **工具调用超时**配置，并在后端 middleware 端到端注入：

* `TOOL_CALL_TIMEOUT_SECONDS` 作为全局持久化配置
* 用户 UI settings 可覆写（并做 clamp/兜底）
* tool calling 相关 metadata 同时包含：
  * `max_tool_calls_per_round`
  * `tool_call_timeout_seconds`

---

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

#### 5.3 Hybrid/BM25/Reranking 重构

* 将配置从“混合概念”拆得更清晰：
  * BM25 权重独立（例如从 `RAG_HYBRID_BM25_WEIGHT` 到 `RAG_BM25_WEIGHT`）
  * BM25 搜索、enriched texts、reranking 分别有独立开关，支持任意混合模式检索
* 新增 **Voyage reranking engine**
  * engine 校验支持 `external` / `voyage`
  * external reranker 的 payload/response 结构更兼容
  * Voyage 包含一套专属的 Tokenizer Model，可以更快地切分和提高嵌入稳定性
    * Tokenizer Model 会进行懒加载和使用一小段文本切分预热，切换 Tokenizer Model 会自动触发 Tokenizer 下载和预热

#### 5.4 对话框一键禁用 RAG（端到端阻断）

新增一个聊天输入区按钮：**Disable RAG**
* 前端：作为 feature 写入请求
* 后端：真正端到端 short-circuit：
  * 不走 model knowledge
  * 不触发 web_search
  * 不处理文件 sources
* 目标：让用户能稳定地把“工具/模型对话”与“RAG 对话”分离开

#### 5.5 Collection 级别的 RAG 独立配置覆写

OUI-X 的一个关键变化：**每个 Knowledge Collection 可以在自身 meta 里保存配置覆写**，从而让不同 collection 使用不同策略（embedding/chunk size/rerank/阈值/top_k 等）。

实现层面：

* 后端检索改为“逐 collection 按 effective config 查询，再 merge 排序”
* UI 增加 collection config modal（用于编辑/保存覆写项）
* config 访问权限对齐：**需要 write 权限**（只读用户不再尝试加载 config，避免 toast 噪音）

#### 5.6 Knowledge Collection Clone（更稳的复制策略）

新增 Knowledge clone API，并持续强化稳定性：

* `POST /knowledge/{id}/clone`
* 优先 direct clone（向量库复制），失败回退 re-embed
* 返回 `warnings`：提示 fallback 原因与部分失败细节
* 对 Chroma：
  * 复制策略升级为更可恢复的复制方式（按 ID copy、指数退避重试）
  * 部分失败文件 fallback re-embed
  * 修复 `has_collection` 兼容不同 Chroma client 返回形态

#### 5.7 对话框文件上传链路改造

对话框上传文件时新增全局开关 **Conversation File Upload Embedding**：

* 开关关闭时，文件不再传入知识库，经过解析后直接注入上下文
* 开关开启时，除图片外的文件在解析后将自动进入知识库，如知识库不存在，将创建一个以用户名+固定后缀名称的知识库
* 知识库存储路径分离：
  * 在 knowledge -> collection 中手动上传文件，或通过对话框上传文件（开关开启），文件实体的存储路径变更到 `./data/vector_db/uploads`
  * 其他文件仍存储在 `./data/uploads`

#### 5.8 文件删除链路改造

* 点击删除 collection 当中的文件或删除 collection 本身时，将会逐一对包含的文件进行判断，如果该文件在其他知识库当中不存在文件共享，那么文件、信息、嵌入数据将在所有链路当中全部移除
  * 如果在其他会话当中存在引用，那么将进行额外确认，如确认删除，那么文件、信息、嵌入数据将在所有链路当中全部移除
  * 如果在其他知识库当中存在文件共享，那么文件实体将被无条件默认保留，其他信息、嵌入数据将全部移除
* 以上所有删除操作都会做一次甚至二次确认

#### 5.9 会话操作选项中新增选项 Add To Collection

* 点击按钮可以将当前会话转换成 Markdown 后保存到指定 Collection 中（走标准知识库入库流程）

### 5) Advanced Params 参数选项改造

* 删除了部分 Ollama 等本地模型专属参数
* 新增 Verbosity、Summary 选项
  * OpenAI 专属参数，映射至 chat completions 及 responses 端点
* Reasoning Effort、Verbosity、Summary 选项改造为滑块+输入框
* 调整了参数顺序
* 将参数 UI 显式统一改为首字母大写格式

---

## 移除了什么？

### 1) 移除 Ollama 相关组件与“内置绑定”

* 去掉 Ollama 相关路由、前端管理组件与脚本
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

---

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



---

## 适用场景建议

* 想要把“工具生态”做成产品核心：优先使用 MCP + native pipeline
* 需要不同知识库集合使用不同检索策略：启用 collection-level config override
* 需要更现代的 OpenAI API 兼容：使用 Responses API 通路
* 希望默认构建更稳定、更轻：使用当前 OUI-X 的默认依赖策略
