"""
GraphRAG 模块：基于 Neo4j 知识图谱的检索增强生成。

职责：
1. 从用户问题中提取医学实体（疾病、症状、药物、科室、检查项目）
2. 将自然语言翻译为 Cypher 查询语句（NL2Cypher），支持错误重试
3. 在 Neo4j 中执行 Cypher，获取图谱关系数据
4. 将图谱查询结果交给 LLM 整合为自然语言回答

典型场景：查询实体间关系，如"高血压有哪些常用药"、"糖尿病的并发症是什么"
"""

from __future__ import annotations
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from src.agents.llm_utils import ainvoke_with_timeout
from langchain_core.language_models import BaseChatModel
from neo4j import AsyncDriver

from src.agents.knowledge.prompts import (
    ENTITY_EXTRACT_PROMPT, NL2CYPHER_PROMPT, GRAPH_QA_PROMPT,
)

MAX_CYPHER_RETRIES = 2  #表示 Cypher 执行失败后最多重试 2 次 ，但加上第1次初始尝试，总共最多执行 3 次 ：


# 实体提取
async def _extract_entities(question: str, llm: BaseChatModel) -> dict:
    prompt = ENTITY_EXTRACT_PROMPT.format(question=question)
    response = await ainvoke_with_timeout(llm, [SystemMessage(content=prompt)], step="knowledge.llm")
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        return {"diseases": [], "symptoms": [], "drugs": [], "departments": [], "checks": []}

# 在 Neo4j 中生成 Cypher 查询语句
async def _generate_cypher(
    question: str, entities: dict, llm: BaseChatModel, error_hint: str = "",
) -> str:
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 Cypher 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2CYPHER_PROMPT.format(
        question=question,
        entities=json.dumps(entities, ensure_ascii=False),
    ) + extra
    response = await ainvoke_with_timeout(llm, [SystemMessage(content=prompt)], step="knowledge.llm")
    cypher = response.content.strip()
    if "```" in cypher:
        cypher = cypher.split("```")[1].lstrip("cypher").strip()
    return cypher

# 执行查询语句
async def _execute_cypher(cypher: str, neo4j_driver: AsyncDriver) -> list[dict]:
    if not cypher:
        return []
    async with neo4j_driver.session() as session:
        result = await session.run(cypher)
        return await result.data()

# 检索原始数据
async def search_graph_raw(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> list[dict]:
    """GraphRAG 检索，返回原始图谱查询结果（不经过 LLM 生成）。"""
    entities = await _extract_entities(question, llm)       # 实体提取
    logger.info(f"GraphRAG 实体提取: {entities}")

    error_hint = ""
    for attempt in range(MAX_CYPHER_RETRIES + 1):
        cypher = await _generate_cypher(question, entities, llm, error_hint)    # 生成 Cypher 查询语句（考虑上一次报错）
        logger.info(f"GraphRAG Cypher (attempt {attempt + 1}): {cypher}")
        try:
            records = await _execute_cypher(cypher, neo4j_driver)    # 执行 Cypher 查询语句
            return records[:20]
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"Cypher 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_CYPHER_RETRIES:
                return []
    return []

# 入口
async def search_graph(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
    role: str = "patient",
    context_text: str = "",
) -> str:
    """GraphRAG 检索 + LLM 生成回答。
    question: 用于实体提取和 Cypher 查询的问题（改写后的问题）
    context_text: 用于最终 LLM 生成回答时的问题（含多轮对话上下文）
    """
    records = await search_graph_raw(question, neo4j_driver, llm)

    if not records:
        return "知识图谱中未找到与您问题相关的信息。"

    # 最终生成回答时用含上下文的问题，让 LLM 理解指代
    answer_question = context_text if context_text else question
    graph_result = json.dumps(records, ensure_ascii=False, indent=2)
    prompt = GRAPH_QA_PROMPT.format(
        question=answer_question, graph_result=graph_result, role=role,
    )
    response = await ainvoke_with_timeout(llm, [SystemMessage(content=prompt)], step="knowledge.llm")
    return response.content