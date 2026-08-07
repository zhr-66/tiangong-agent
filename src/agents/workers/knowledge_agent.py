from __future__ import annotations

from langchain.agents import create_agent
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

from src.agents.knowledge.audit import QueryAuditLog, Timer
from src.agents.knowledge.conversation import (
    append_turn,
    format_context,
    load_conversation_context,
)
from src.agents.knowledge.tools import KnowledgeDeps, build_knowledge_tools
from src.core.config import get_llm, get_settings
from src.infra.redis_cache import get_redis_client

settings = get_settings()


KNOWLEDGE_SYSTEM_PROMPT = """你是天宫医疗的知识问答助手，面向患者、医生、药师和企业内部员工提供专业知识服务。

## 你的工具

- search_knowledge_docs：查询临床指南、药品说明书、医院制度和医学文献。
- search_knowledge_graph：查询疾病、症状、药物、科室和检查之间的关系。
- search_knowledge_sql：查询问诊量、库存、排名和趋势等运营统计数据。
- search_knowledge_multi：融合文档和知识图谱，适合需要多个来源的复杂问题。
- review_prescription_tool：审核处方的剂量、配伍禁忌、过敏冲突和重复用药。

## 工具选择策略

1. 处方审核、配伍禁忌、过敏冲突或用药安全校验，必须调用 review_prescription_tool。
2. 指南、说明书、病因、预防或治疗原则，调用 search_knowledge_docs。
3. 疾病与症状、药物、科室或检查的关系，调用 search_knowledge_graph。
4. 统计、数量、排名、趋势或库存，调用 search_knowledge_sql。
5. 同时需要文档依据与图谱关系，或无法确定单一来源时，优先调用 search_knowledge_multi。

## 工作原则

- 医学或运营事实必须先调用合适的工具获得证据，再回答用户。
- 只基于工具返回的结果回答；证据不足时明确说明。
- 涉及用药安全时提醒用户遵医嘱，并建议向医生或药师确认。
- 不要向用户暴露工具、数据库或内部工作流细节。
"""


def _message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def create_knowledge_agent(deps: KnowledgeDeps):
    """Build a request-scoped Tool Calling Agent bound to the current user context."""
    return create_agent(
        model=deps.llm,
        tools=build_knowledge_tools(deps),
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )


class KnowledgeAgent:
    """Reusable service that assembles context-bound knowledge tools per request."""

    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.milvus_client = None
        self.neo4j_driver = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        self.llm = get_llm(temperature=0.3)
        self.embedding_model = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )
        self.milvus_client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        )
        self._initialized = True

    async def _get_neo4j_driver(self):
        if self.neo4j_driver is None:
            from src.infra.neo4j_client import get_neo4j_driver

            self.neo4j_driver = get_neo4j_driver()
        return self.neo4j_driver

    async def query(
        self,
        question: str,
        role: str = "patient",
        user_id: str = "",
        session_id: str = "",
        db_session=None,
    ) -> str:
        if not self._initialized:
            await self.initialize()

        redis_client = await get_redis_client()
        history = await load_conversation_context(redis_client, user_id, session_id)
        context_text = format_context(history)
        agent_input = (
            f"{context_text}\n\n当前问题：{question}"
            if context_text
            else question
        )

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

        with Timer() as timer:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": agent_input}]}
            )

        messages = result.get("messages") or []
        answer = _message_content(messages[-1].content).strip() if messages else ""
        if not answer:
            answer = "暂时未能生成有效回答，请稍后重试。"

        if user_id and session_id:
            await append_turn(redis_client, user_id, session_id, question, answer)

        QueryAuditLog.log(
            user_id=user_id or "anonymous",
            role=role,
            question=question,
            intent="tool_calling",
            channels=[],
            answer_preview=answer,
            duration_ms=timer.elapsed_ms,
        )
        return answer


_knowledge_agent: KnowledgeAgent | None = None


async def get_knowledge_agent() -> KnowledgeAgent:
    global _knowledge_agent
    if _knowledge_agent is None:
        _knowledge_agent = KnowledgeAgent()
    return _knowledge_agent
