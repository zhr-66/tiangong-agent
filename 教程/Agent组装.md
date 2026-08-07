# Step 17：Agent 组装

本项目的知识服务由 Supervisor 调度到三个 Worker：知识问答 Agent、药物咨询 Agent 和运营数据 Agent。三者复用同一套知识工具，但按职责分配不同的工具权限。

```text
Supervisor / FastAPI
        ↓
KnowledgeAgent.query(...)
        ↓
当前请求的 KnowledgeDeps
        ↓
LangChain Tool Calling Agent
        ↓
DocRAG / GraphRAG / NL2SQL / Fusion / 处方审核
```

## 1. 知识 Agent：请求作用域的 Tool Calling Agent

代码位置：[src/agents/workers/knowledge_agent.py](../src/agents/workers/knowledge_agent.py)

知识 Agent 对外仍通过 `get_knowledge_agent().query(...)` 调用。`KnowledgeAgent` 是轻量单例服务，用于复用 LLM、Embedding、Milvus 和 Neo4j 客户端；但它不会缓存带工具的 Agent。

原因是工具通过 `KnowledgeDeps` 闭包持有当前请求的 `user_id`、`role` 和 `db_session`。如果缓存工具 Agent，后一个请求可能意外复用前一个用户的上下文。

每个请求都按当前依赖重新组装：

```python
def create_knowledge_agent(deps: KnowledgeDeps):
    return create_agent(
        model=deps.llm,
        tools=build_knowledge_tools(deps),
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )
```

调用时会创建当前请求的依赖容器：

```python
deps = KnowledgeDeps(
    llm=self.llm,
    embedding_model=self.embedding_model,
    milvus_client=self.milvus_client,
    neo4j_driver=await self._get_neo4j_driver(),
    db_session=db_session,
    user_id=user_id or "anonymous",
    role=role,
)
agent = create_knowledge_agent(deps)
```

随后通过 `agent.ainvoke(...)` 将问题交给模型。模型通过 `KNOWLEDGE_SYSTEM_PROMPT` 理解工具选择策略，而不是由 Python 关键词规则强制决定 RAG 类型。

知识 Agent 拥有五个工具：

| 工具 | 用途 |
|---|---|
| `search_knowledge_docs` | 临床指南、药品说明书、制度和医学文献 |
| `search_knowledge_graph` | 疾病、症状、药物、科室和检查之间的图谱关系 |
| `search_knowledge_sql` | 问诊量、库存、排名和趋势等运营统计 |
| `search_knowledge_multi` | 文档与图谱的按需融合检索 |
| `review_prescription_tool` | 剂量、配伍、过敏和重复用药审核 |

提示词中的关键约束：

```text
处方审核 / 配伍禁忌 / 过敏冲突
→ 必须调用 review_prescription_tool

单一文档知识
→ search_knowledge_docs

实体关系
→ search_knowledge_graph

统计数据
→ search_knowledge_sql

复杂问题或来源不确定
→ 优先 search_knowledge_multi
```

调用前，Agent 从 Redis 读取 `knowledge_ctx:{user_id}:{session_id}` 的历史问答并拼入当前输入；调用后把原问题和最终回答写回 Redis，同时写入审计日志。

## 2. 工具封装与 SQL 会话回退

代码位置：[src/agents/knowledge/tools.py](../src/agents/knowledge/tools.py)

所有检索能力先被封装成 LangChain `@tool`。工具层负责 Query Rewrite、调用计时和审计记录，具体 RAG 模块只负责检索与生成。

`KnowledgeDeps` 支持请求注入的数据库会话，也支持可注入的会话工厂：

```python
class KnowledgeDeps:
    def __init__(
        self,
        llm,
        embedding_model,
        milvus_client,
        neo4j_driver,
        db_session=None,
        user_id="anonymous",
        role="patient",
        session_factory=None,
    ):
        ...
```

SQL 工具的会话策略：

```python
@asynccontextmanager
async def _sql_session(deps: KnowledgeDeps):
    if deps.db_session is not None:
        yield deps.db_session
        return

    session_factory = deps.session_factory or AsyncSessionLocal
    async with session_factory() as session:
        yield session
```

因此：

```text
FastAPI 直连知识问答
→ 复用 Depends(get_db) 注入的会话

Supervisor / Operation Agent 调用
→ 工具自行创建并关闭本次查询会话
```

这解决了 Operation Agent 在没有 FastAPI `db_session` 时无法执行 NL2SQL 的问题。SQL 的只读 `SELECT` 校验、敏感字段黑名单、`LIMIT` 和超时限制仍由 [src/agents/knowledge/nl2sql.py](../src/agents/knowledge/nl2sql.py) 负责。

多通道工具也会规范化结果：

```python
answer, channels = await multi_channel_search(...)
return answer
```

`multi_channel_search` 原始返回 `(answer, channels)`；Tool Calling Agent 只能接收用户可读字符串，因此工具层只返回 `answer`，并将实际执行的 `channels` 写入审计日志。

## 3. 药物 Agent：四工具白名单

代码位置：[src/agents/workers/drug_agent.py](../src/agents/workers/drug_agent.py)

药物 Agent 不具备运营 SQL 权限，只能使用：

```python
allowed_tool_names = {
    "search_knowledge_docs",
    "search_knowledge_graph",
    "search_knowledge_multi",
    "review_prescription_tool",
}
tools = [
    tool for tool in build_knowledge_tools(deps)
    if tool.name in allowed_tool_names
]
```

这里使用显式白名单，而不是“排除 SQL 工具”。即使将来知识工具集新增内部工具，也不会被药物 Agent 意外获得。

场景映射：

```text
药品说明书、不良反应、用法用量
→ search_knowledge_docs

疾病常用药、药物关系
→ search_knowledge_graph

合并症与用药禁忌
→ search_knowledge_multi

处方、配伍、剂量、过敏审核
→ review_prescription_tool
```

## 4. 运营 Agent：SQL-only

代码位置：[src/agents/workers/operation_agent.py](../src/agents/workers/operation_agent.py)

运营 Agent 只保留 `search_knowledge_sql`：

```python
tools = [
    tool for tool in build_knowledge_tools(deps)
    if tool.name == "search_knowledge_sql"
]
```

它只能处理问诊量、药品库存、科室排名和趋势等运营统计；禁止将文档、图谱或处方工具暴露给运营 Agent。实际 SQL 安全还依赖数据库账号权限和 `nl2sql.py` 的查询校验，不能仅依赖系统提示词。

## 5. 验证方式

执行离线测试，不要求启动 Redis、Milvus、Neo4j 或 PostgreSQL：

```bash
python -m pytest test/test_agent_assembly.py test/test_fast_routing.py test/test_inquiry_scoring.py -q
```

测试覆盖：

- 请求注入会话优先于新建会话；新建会话会在调用后关闭。
- 多通道工具始终返回字符串，并记录实际执行通道。
- 知识 Agent 组装出五个教程规定的工具，并绑定当前用户、角色和数据库会话。
- 药物 Agent 只能使用四个批准工具；运营 Agent 只能使用 SQL 工具。
