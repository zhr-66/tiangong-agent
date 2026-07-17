"""
多通道融合检索模块。

职责：
1. 两阶段检索：先查主导通道（doc_rag），LLM 判断证据是否充分
2. 证据不足时按需补查（graph_rag / sql），最多补查一次
3. 汇总各通道结果，交给 LLM 融合生成统一回答
4. 对融合回答做幻觉检测，不达标时附加风险提示

设计原则（克制）：
- 外层 route=multi 仍由显式路由决定，本模块不改变路由决策
- LLM 只判断"证据够不够"，不选通道、不循环调用
- 最多补查一次，避免退化成 Agent 循环

典型场景：复杂问题需要同时查多个来源才能回答
"""

from __future__ import annotations
import asyncio
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from src.agents.llm_utils import agenerate_final, ainvoke_with_timeout
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.knowledge.prompts import FUSION_PROMPT, EVIDENCE_SUFFICIENCY_PROMPT
from src.agents.knowledge.doc_rag import search_docs_raw, format_doc_context
from src.agents.knowledge.graph_rag import search_graph_raw
from src.agents.knowledge.hallucination_check import check_hallucination


def _split_subquestions(question: str) -> list[str]:
    """用简单规则拆解子问题（按连词分割）"""
    separators = ["和", "及", "并", "以及", "与", "、"]
    parts = [question]
    for sep in separators:
        new_parts = []
        for p in parts:
            new_parts.extend([s.strip() for s in p.split(sep)])
        parts = new_parts
    return [p for p in parts if p and len(p) > 2]


async def _check_evidence_sufficiency(
    question: str,
    evidence: str,
    llm: BaseChatModel,
) -> dict:
    """
    让 LLM 判断当前检索结果是否足以回答问题。
    返回 {"sufficient": bool, "missing_aspect": "graph_rag|sql|null", "reason": str}
    
    策略：
    1. 代码拆解问题为子需求（如"病因和常用药"→["病因", "常用药"]）
    2. 对每个子需求逐一检查
    3. 任一子需求未覆盖即判 insufficient
    """
    prompt = EVIDENCE_SUFFICIENCY_PROMPT.format(question=question, evidence=evidence[:1500])
    try:
        response = await ainvoke_with_timeout(
            llm, [SystemMessage(content=prompt)], step="knowledge.evidence_check",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        
        # 规范化 missing_aspect
        missing = result.get("missing_aspect")
        if missing not in ("graph_rag", "sql", None):
            missing = None
            result["missing_aspect"] = None
        
        # 代码层面二次校验：检查 LLM 是否覆盖了所有子需求
        subquestions = _split_subquestions(question)
        reason = result.get("reason", "")
        
        # 如果问题被拆解成多个子需求，但 LLM 的 reason 只提到了一部分
        if len(subquestions) > 1:
            # 检查每个子需求是否都被提到
            covered_subs = []
            uncovered_subs = []
            for sub in subquestions:
                # 简单检查：子需求关键词是否在 reason 中出现
                sub_keywords = sub[:4]  # 取前4个字符作为关键词
                if sub_keywords in reason or sub[:3] in reason:
                    covered_subs.append(sub)
                else:
                    uncovered_subs.append(sub)
            
            if uncovered_subs:
                # LLM 漏了某些子需求，强制判 insufficient
                result["sufficient"] = False
                if missing is None:
                    # 猜测缺失的通道：如果漏的是"常用药/推荐药/症状/检查"→graph_rag
                    uncovered_str = "".join(uncovered_subs)
                    if any(k in uncovered_str for k in ["常用药", "推荐药", "症状", "检查", "并发症"]):
                        result["missing_aspect"] = "graph_rag"
                    elif any(k in uncovered_str for k in ["统计", "数量", "排名", "库存"]):
                        result["missing_aspect"] = "sql"
                    else:
                        result["missing_aspect"] = "graph_rag"  # 默认补查图谱
                result["reason"] = f"LLM漏检子需求: {uncovered_subs}; 原reason: {reason}"
                logger.info(f"证据充分性强制修正: 漏检子需求={uncovered_subs}, reason={result['reason']}")
            else:
                result["sufficient"] = bool(result.get("sufficient", True))
        else:
            # 只有一个子需求，信任 LLM 的判断
            if missing is not None:
                result["sufficient"] = False
            else:
                result["sufficient"] = bool(result.get("sufficient", True))
        
        return result
    except Exception as e:
        logger.warning(f"证据充分性判断失败: {e}")
        return {"sufficient": True, "missing_aspect": None, "reason": "check failed, skip补查"}


def _format_sources(
    doc_hits: list[dict] | None,
    graph_records: list[dict] | None,
    sql_answer: str | None,
) -> tuple[str, list[str]]:
    """汇总各通道结果为 LLM 可读的 sources 文本 + evidence 摘要列表。"""
    source_parts = []
    evidence_parts = []

    if doc_hits:
        ctx = format_doc_context(doc_hits)
        source_parts.append(f"### 文档检索结果\n{ctx}")
        evidence_parts.append(ctx[:1000])

    if graph_records:
        graph_str = json.dumps(graph_records, ensure_ascii=False, indent=2)
        source_parts.append(f"### 知识图谱检索结果\n{graph_str}")
        evidence_parts.append(graph_str[:1000])

    if sql_answer and isinstance(sql_answer, str):
        source_parts.append(f"### 运营数据查询结果\n{sql_answer}")
        evidence_parts.append(sql_answer[:1000])

    return "\n\n".join(source_parts), evidence_parts


async def multi_channel_search(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    neo4j_driver: AsyncDriver,
    db_session: AsyncSession | None = None,
    channels: list[str] | None = None,
    role: str = "patient",
    context_text: str = "",
    sub_queries: list[str] | None = None,
) -> str:
    """
    两阶段多通道检索 → 结果融合 → 幻觉检测 → 返回最终回答。

    流程：
    1. 阶段一：查主导通道（doc_rag，通常最快最稳）
    2. LLM 判断证据是否充分（一次轻量调用）
    3. 阶段二：不足时按 missing_aspect 补查（graph_rag 或 sql），最多一次
    4. 融合所有结果 → LLM 生成回答 → 幻觉检测

    question: 用于检索的问题（改写后的问题）
    context_text: 用于最终 LLM 生成回答的问题（含多轮对话上下文）
    channels: 指定使用哪些通道 ["doc_rag", "graph_rag", "nl2sql"]，默认 doc_rag + graph_rag
    """
    if channels is None:
        channels = ["doc_rag", "graph_rag"]

    # 构建检索用的查询列表：有子查询时用子查询分别检索，否则用单一 question
    search_queries = sub_queries if sub_queries and len(sub_queries) > 1 else [question]

    # ---------- 阶段一：查主导通道 ----------
    # doc_rag 是默认主导通道；如果调用方明确不要 doc_rag，则用 graph_rag 作主导
    primary_channel = "doc_rag" if "doc_rag" in channels else (
        "graph_rag" if "graph_rag" in channels else "nl2sql"
    )

    doc_hits: list[dict] | None = None
    graph_records: list[dict] | None = None
    sql_answer: str | None = None

    logger.info(f"multi_channel_search 阶段一: 主导通道={primary_channel} | queries={search_queries}")

    try:
        if primary_channel == "doc_rag":
            # 对每个子查询分别检索，合并去重
            all_hits = []
            seen_ids = set()
            for q in search_queries:
                hits = await search_docs_raw(q, embedding_model, milvus_client)
                if hits:
                    for h in hits:
                        hit_id = h.get("id") or h.get("content", "")[:80]
                        if hit_id not in seen_ids:
                            seen_ids.add(hit_id)
                            all_hits.append(h)
            doc_hits = all_hits if all_hits else None
        elif primary_channel == "graph_rag":
            all_records = []
            for q in search_queries:
                records = await search_graph_raw(q, neo4j_driver, llm)
                if records:
                    all_records.extend(records)
            graph_records = all_records if all_records else None
        elif primary_channel == "nl2sql" and db_session:
            from src.agents.knowledge.nl2sql import search_sql
            sql_answer = await search_sql(question, llm, db_session)
    except Exception as e:
        logger.warning(f"主导通道 {primary_channel} 检索失败: {e}")

    # ---------- 判断证据是否充分 ----------
    sources, evidence_parts = _format_sources(doc_hits, graph_records, sql_answer)
    if not sources:
        # 主导通道啥都没拿到，直接按配置补查剩余通道（不走 LLM 判断）
        logger.info("主导通道无结果，直接补查剩余通道")
        missing = "graph_rag" if primary_channel != "graph_rag" and "graph_rag" in channels else (
            "sql" if primary_channel != "nl2sql" and "nl2sql" in channels and db_session else None
        )
    else:
        sufficiency = await _check_evidence_sufficiency(question, sources, llm)
        missing = None if sufficiency["sufficient"] else sufficiency.get("missing_aspect")
        logger.info(
            "证据充分性判断: sufficient={} missing={} reason={}",
            sufficiency["sufficient"], missing, sufficiency.get("reason", ""),
        )
        # 补查目标不在 channels 白名单里，则不补查
        if missing == "graph_rag" and "graph_rag" not in channels:
            missing = None
        elif missing == "sql" and ("nl2sql" not in channels or not db_session):
            missing = None

    # ---------- 阶段二：按需补查（最多一次） ----------
    if missing == "graph_rag" and graph_records is None:
        logger.info("阶段二: 补查 graph_rag")
        try:
            all_records = []
            for q in search_queries:
                records = await search_graph_raw(q, neo4j_driver, llm)
                if records:
                    all_records.extend(records)
            graph_records = all_records if all_records else None
        except Exception as e:
            logger.warning(f"补查 graph_rag 失败: {e}")
    elif missing == "sql" and sql_answer is None and db_session:
        logger.info("阶段二: 补查 nl2sql")
        try:
            from src.agents.knowledge.nl2sql import search_sql
            sql_answer = await search_sql(question, llm, db_session)
        except Exception as e:
            logger.warning(f"补查 nl2sql 失败: {e}")

    # ---------- 融合生成 ----------
    sources, evidence_parts = _format_sources(doc_hits, graph_records, sql_answer)
    if not sources:
        return "所有检索通道均未找到与您问题相关的信息。", []

    answer_question = context_text if context_text else question
    prompt = FUSION_PROMPT.format(
        question=answer_question, sources=sources, role=role,
    )
    # 最终面向用户的生成：流式接口下逐 token 推送
    # （之后的幻觉检测只会在末尾追加警告，不修改正文，由路由层 replace 事件对账）
    response = await agenerate_final(llm, [SystemMessage(content=prompt)], step="knowledge.llm")
    answer = response.content

    # ---------- 幻觉检测 ----------
    evidence = "\n".join(evidence_parts)
    hal_result = await check_hallucination(question, evidence, answer, llm)
    if not hal_result["is_grounded"]:
        claims = "、".join(hal_result.get("unsupported_claims", []))
        answer += f"\n\n⚠️ 提示：以下内容未在检索结果中找到充分依据，请谨慎参考：{claims}"

    # 实际执行的通道
    executed_channels = []
    if doc_hits:
        executed_channels.append("doc_rag")
    if graph_records:
        executed_channels.append("graph_rag")
    if sql_answer:
        executed_channels.append("nl2sql")

    return answer, executed_channels
