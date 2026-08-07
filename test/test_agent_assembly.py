from contextlib import asynccontextmanager

import pytest

from src.agents.knowledge.tools import KnowledgeDeps, _sql_session, build_knowledge_tools


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
