from __future__ import annotations
import json
from loguru import logger
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient
from langchain_core.messages import SystemMessage

from src.core.config import get_settings
from src.infra.redis_cache import get_redis_client
from src.agents.knowledge import (
    rewrite_query, ROUTE_PROMPT,
    search_docs, search_graph, search_sql,
    multi_channel_search, review_prescription,
    QueryAuditLog, Timer,
    load_conversation_context, format_context, append_turn,
)

settings = get_settings()


def _looks_like_no_context_answer(answer: str) -> bool:
    return "未找到" in answer or "not found" in answer.lower()


class KnowledgeAgent:
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.milvus_client = None
        self.neo4j_driver = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        try:
            self.llm = await self._get_llm()
            self.embedding_model = DashScopeEmbeddings(
                model=settings.EMBEDDING_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            self.milvus_client = MilvusClient(
                uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            )
            self._initialized = True
            logger.info("KnowledgeAgent initialized")
        except Exception as e:
            logger.error(f"KnowledgeAgent initialization failed: {e}")
            raise

    async def _get_llm(self):
        from src.core.config import get_llm
        return get_llm(temperature=0.3)

    async def _get_neo4j_driver(self):
        if self.neo4j_driver is None:
            from src.infra.neo4j_client import get_neo4j_driver
            self.neo4j_driver = get_neo4j_driver()
        return self.neo4j_driver

    async def _route_query(self, question: str) -> dict:
        prompt = ROUTE_PROMPT.format(question=question)
        try:
            response = await self.llm.ainvoke([SystemMessage(content=prompt)]) #判断用户问题应该使用哪种检索方式
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1].lstrip("json").strip()
            result = json.loads(content)
            return result
        except Exception as e:
            logger.warning(f"Knowledge route failed: {e}")
            return {"route": "doc_rag", "reason": "route failed, fallback to doc_rag"}

    async def _get_db_session(self):
        from src.infra.database import get_db

        async for db in get_db():
            return db

    async def _search_docs_with_graph_fallback(self, question: str, role: str) -> tuple[str, list[str]]:
        answer = await search_docs(
            question, self.embedding_model, self.milvus_client,
            self.llm, role=role, use_hyde=True,
        )
        channels = ["doc_rag"]

        if _looks_like_no_context_answer(answer):
            logger.info("DocRAG returned no context; fallback to GraphRAG")
            neo4j_driver = await self._get_neo4j_driver()
            graph_answer = await search_graph(question, neo4j_driver, self.llm, role)
            if not _looks_like_no_context_answer(graph_answer):
                return graph_answer, ["doc_rag", "graph_rag"]

        return answer, channels

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

        with Timer() as timer:
            # 加载多轮对话上下文
            redis_client = await get_redis_client()
            history = await load_conversation_context(redis_client, user_id, session_id)
            context_text = format_context(history)

            # 有历史上下文时，增强问题以便 LLM 理解指代关系
            if context_text:
                enriched_question = f"{context_text}\n\n当前问题：{question}"
            else:
                enriched_question = question

            rewrite_result = await rewrite_query(enriched_question, self.llm, role)  # 改写口语
            intent = rewrite_result.get("intent", "knowledge_qa")

            route_result = await self._route_query(enriched_question) #判断用户问题应该使用哪种检索方式
            route = route_result.get("route", "doc_rag")

            logger.info(
                "Knowledge query | question={} | route={} | intent={}",
                question[:60], route, intent,
            )

            if route == "prescription":
                neo4j_driver = await self._get_neo4j_driver()
                answer = await review_prescription(
                    enriched_question, self.llm, self.embedding_model,
                    self.milvus_client, neo4j_driver,
                )
                channels = ["prescription"]

            elif route == "graph_rag":
                neo4j_driver = await self._get_neo4j_driver()
                answer = await search_graph(enriched_question, neo4j_driver, self.llm, role)
                channels = ["graph_rag"]

            elif route == "nl2sql":
                if db_session is None:
                    db_session = await self._get_db_session()
                answer = await search_sql(enriched_question, self.llm, db_session)
                channels = ["nl2sql"]

            elif route == "multi":
                neo4j_driver = await self._get_neo4j_driver()
                if db_session is None:
                    db_session = await self._get_db_session()
                answer = await multi_channel_search(    # 多通道检索
                    enriched_question, self.llm, self.embedding_model,
                    self.milvus_client, neo4j_driver, db_session,
                    channels=["doc_rag", "graph_rag"],
                    role=role,
                )
                channels = ["doc_rag", "graph_rag"]

            else:
                answer, channels = await self._search_docs_with_graph_fallback(enriched_question, role)

            # 保存本轮对话到上下文
            if user_id and session_id:
                await append_turn(redis_client, user_id, session_id, question, answer)

            QueryAuditLog.log(
                user_id=user_id,
                role=role,
                question=question,
                intent=intent,
                channels=channels,
                answer_preview=answer,
                duration_ms=timer.elapsed_ms,
            )

        return answer


_knowledge_agent = None


async def get_knowledge_agent() -> KnowledgeAgent:
    global _knowledge_agent
    if _knowledge_agent is None:
        _knowledge_agent = KnowledgeAgent()
        await _knowledge_agent.initialize()
    return _knowledge_agent
