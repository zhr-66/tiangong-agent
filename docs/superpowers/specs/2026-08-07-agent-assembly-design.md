# Agent 组装设计

## 目标

将知识问答、药物咨询和运营数据三个 Worker 组装为可调用工具的 LangChain Agent，同时保持现有 FastAPI、Supervisor 和 Redis 会话接口不变。

## 当前问题

- `KnowledgeAgent` 目前通过内部 LLM 路由直接调用检索函数，未按教程组装为包含五个知识工具的 Tool Calling Agent。
- `search_knowledge_multi` 将 `multi_channel_search` 的 `(answer, channels)` 元组直接作为工具结果返回，违反工具应返回用户可读字符串的约定。
- `OperationAgent` 的 SQL 工具依赖外部注入的数据库会话；Supervisor 调用路径没有该会话，导致运营查询不可用。

## 方案

### 保持外部接口

保留 `get_knowledge_agent()` 和 `KnowledgeAgent.query(question, role, user_id, session_id, db_session)`。FastAPI 的直连知识问答路径和 Supervisor 的 `call_knowledge_agent` 无需改变调用名称或参数。

### 每次查询组装知识 Agent

`KnowledgeAgent` 继续作为轻量单例服务，复用 LLM、Embedding、Milvus 和 Neo4j 客户端。每次 `query` 根据当前 `user_id`、`role` 和可选 `db_session` 创建 `KnowledgeDeps`，调用 `build_knowledge_tools(deps)` 后使用 `create_agent` 组装一个请求作用域的 Tool Calling Agent。

这是必要的：工具由闭包持有 `KnowledgeDeps`，全局缓存的工具会错误复用前一用户的身份、角色或会话。

该 Agent 使用 `KNOWLEDGE_SYSTEM_PROMPT`，要求优先使用工具并按问题选择：

- `search_knowledge_docs`
- `search_knowledge_graph`
- `search_knowledge_sql`
- `search_knowledge_multi`
- `review_prescription_tool`

调用前加载 Redis 知识问答历史并作为上下文消息；调用后提取最后一条 AI 回复、写回历史并写入审计日志。

### 工具层修复

`search_knowledge_multi` 解包 `multi_channel_search` 的返回值，只返回 `answer` 字符串，并将实际 `channels` 写入审计日志。

SQL 工具优先使用请求传入的 `AsyncSession`。如果没有传入，工具用 `AsyncSessionLocal` 创建仅覆盖本次工具调用的会话，并在调用结束后关闭。该策略使 Supervisor、Drug Agent 和 Operation Agent 都能安全执行只读 SQL；SQL 本身仍由现有校验器限制为 `SELECT`、黑名单字段和 `LIMIT`。

### Worker 权限

- Knowledge Agent：五个工具全部可用。
- Drug Agent：保留文档、图谱、多通道和处方审核工具，排除 SQL 工具。
- Operation Agent：只保留 SQL 工具。

现有 Worker 的系统提示词继续作为工具选择和安全约束，不改变 Supervisor 暴露的 `call_drug_agent`、`call_operation_agent` 等工具。

## 错误处理

- 工具底层已有检索、模型和数据库异常降级行为，组装层不吞掉用户可读错误。
- 运营 SQL 会话在 `finally` 语义的 async context manager 中释放。
- 多通道融合无证据时，工具返回融合模块已有的用户可读提示，而非 Python 元组或异常对象。

## 测试

新增离线单元测试，不调用 LLM、Milvus、Neo4j 或 PostgreSQL：

1. 知识 Agent 工厂组装出教程要求的五个工具，并使用知识 Agent 系统提示词。
2. Drug Agent 的工具集不包含 SQL；Operation Agent 的工具集只包含 SQL。
3. 多通道工具解包融合结果并返回字符串，同时记录实际通道。
4. SQL 工具在未传入会话时通过短生命周期会话工厂执行，而传入会话时复用该会话。

测试使用依赖注入或局部替身隔离外部服务，只验证项目自身的组装与会话管理行为。

## 非目标

- 不重写 DocRAG、GraphRAG、NL2SQL、Fusion 或处方审核的检索逻辑。
- 不改变 Supervisor 的路由策略、FastAPI 接口或 Redis 键格式。
- 不增加真实挂号、认证或新的外部基础设施。
