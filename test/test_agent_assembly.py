from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.agents.knowledge.tools import KnowledgeDeps, _sql_session, build_knowledge_tools
from src.agents.workers import knowledge_agent


@pytest.fixture
def deps_factory():
    def factory(**overrides):
        return KnowledgeDeps(
            llm=overrides.pop("llm", object()),
            embedding_model=overrides.pop("embedding_model", object()),
            milvus_client=overrides.pop("milvus_client", object()),
            neo4j_driver=overrides.pop("neo4j_driver", object()),
            **overrides,
        )

    return factory


@pytest.mark.asyncio
async def test_sql_session_reuses_request_session(deps_factory):
    request_session = object()
    deps = deps_factory(db_session=request_session)

    async with _sql_session(deps) as session:
        assert session is request_session


@pytest.mark.asyncio
async def test_sql_session_uses_and_closes_factory_when_request_has_none(deps_factory):
    lifecycle = []

    @asynccontextmanager
    async def factory():
        lifecycle.append("entered")
        yield "created-session"
        lifecycle.append("closed")

    deps = deps_factory(db_session=None, session_factory=factory)
    async with _sql_session(deps) as session:
        assert session == "created-session"
    assert lifecycle == ["entered", "closed"]


@pytest.mark.asyncio
async def test_multi_tool_returns_answer_string_and_audits_actual_channels(
    monkeypatch, deps_factory
):
    async def fake_rewrite(question, llm, role):
        return {"queries": [question], "intent": "knowledge_qa"}

    async def fake_multi(**kwargs):
        return "融合答案", ["doc_rag", "graph_rag"]

    audit_calls = []
    monkeypatch.setattr("src.agents.knowledge.query_rewriter.rewrite_query", fake_rewrite)
    monkeypatch.setattr("src.agents.knowledge.fusion.multi_channel_search", fake_multi)
    monkeypatch.setattr(
        "src.agents.knowledge.tools.QueryAuditLog.log",
        lambda *args, **kwargs: audit_calls.append(args),
    )

    multi_tool = next(
        tool
        for tool in build_knowledge_tools(deps_factory())
        if tool.name == "search_knowledge_multi"
    )

    assert await multi_tool.ainvoke({"question": "复杂问题"}) == "融合答案"
    assert audit_calls[0][4] == ["doc_rag", "graph_rag"]


def test_create_knowledge_agent_exposes_all_tutorial_tools(monkeypatch, deps_factory):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "assembled-agent"

    monkeypatch.setattr(knowledge_agent, "create_agent", fake_create_agent)
    result = knowledge_agent.create_knowledge_agent(deps_factory())

    assert result == "assembled-agent"
    assert {tool.name for tool in captured["tools"]} == {
        "search_knowledge_docs",
        "search_knowledge_graph",
        "search_knowledge_sql",
        "search_knowledge_multi",
        "review_prescription_tool",
    }
    assert captured["system_prompt"] == knowledge_agent.KNOWLEDGE_SYSTEM_PROMPT
    assert "review_prescription_tool" in captured["system_prompt"]


@pytest.mark.asyncio
async def test_query_builds_agent_with_current_request_context(monkeypatch):
    created_deps = []

    class FakeRedis:
        async def get(self, key):
            return None

    class FakeAgent:
        async def ainvoke(self, payload):
            return {"messages": [SimpleNamespace(content="工具回答")]}

    async def fake_redis():
        return FakeRedis()

    async def fake_append_turn(*args):
        return []

    def fake_create(deps):
        created_deps.append(deps)
        return FakeAgent()

    monkeypatch.setattr(knowledge_agent, "get_redis_client", fake_redis)
    monkeypatch.setattr(knowledge_agent, "create_knowledge_agent", fake_create)
    monkeypatch.setattr(knowledge_agent, "append_turn", fake_append_turn)

    service = knowledge_agent.KnowledgeAgent()
    service.llm = object()
    service.embedding_model = object()
    service.milvus_client = object()
    service.neo4j_driver = object()
    service._initialized = True

    reply = await service.query(
        "高血压常用药", role="doctor", user_id="u1", session_id="s1", db_session="db"
    )

    assert reply == "工具回答"
    assert created_deps[0].user_id == "u1"
    assert created_deps[0].role == "doctor"
    assert created_deps[0].db_session == "db"
