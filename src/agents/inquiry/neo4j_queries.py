# src/agents/inquiry/neo4j_queries.py

from __future__ import annotations
from loguru import logger
from neo4j import AsyncDriver

from src.core.config import get_settings
from src.agents.inquiry.scoring import score_candidates
from src.agents.inquiry.state import CandidateDisease

# Cypher 粗召回上限：先按命中数取前 N 个候选，再由 Python 侧精确打分排序。
# 用户症状通常 2~5 个，共享任一症状的疾病可能上百，50 足够覆盖真实候选。
_RECALL_LIMIT = 50


async def query_candidate_diseases(
    confirmed_symptoms: list[str],
    neo4j_driver: AsyncDriver,
    top_k: int = 10,
) -> list[CandidateDisease]:
    """
    根据已确认症状列表，从 Neo4j 查询候选疾病。

    分两步：
    1. Cypher 粗召回：取共享症状最多的前 _RECALL_LIMIT 个疾病及其症状/df 数据
    2. Python 精排：scoring.score_candidates 按 INQUIRY_SCORING 配置打分
       （idf_f1 = IDF 加权双向 F1；legacy = 命中数/总症状数）
    打分逻辑与离线评估台（scripts/eval_confidence.py）共用同一实现。
    """
    if not confirmed_symptoms:
        return []

    cypher = """
    MATCH (n:Disease) WITH count(n) AS n_diseases
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name IN $confirmed_symptoms
    WITH n_diseases, d, collect(s.name) AS matched_symptoms, count(s) AS matched_count
    ORDER BY matched_count DESC
    LIMIT $recall_limit
    MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
    RETURN
        n_diseases,
        d.name AS disease,
        matched_symptoms,
        collect(all_s.name) AS all_symptoms,
        collect({name: all_s.name, df: all_s.df}) AS symptom_dfs
    """

    async with neo4j_driver.session() as session:
        result = await session.run(
            cypher,
            confirmed_symptoms=confirmed_symptoms,
            recall_limit=_RECALL_LIMIT,
        )
        records = await result.data()

    if not records:
        return []

    n_diseases = records[0]["n_diseases"]

    # 汇总本次涉及症状的 df 表（df 为 None 表示统计未跑，scoring 会安全回退 legacy）
    symptom_df: dict[str, int] = {}
    for r in records:
        for item in r["symptom_dfs"]:
            if item["df"] is not None:
                symptom_df[item["name"]] = item["df"]

    settings = get_settings()
    if settings.INQUIRY_SCORING == "idf_f1" and not symptom_df:
        logger.warning(
            "Symptom.df 统计缺失，回退 legacy 打分。"
            "请执行: python scripts/init_neo4j.py --stats-only"
        )

    raw = [
        {
            "name": r["disease"],
            "matched_symptoms": r["matched_symptoms"],
            "all_symptoms": r["all_symptoms"],
        }
        for r in records
    ]
    scored = score_candidates(
        raw, confirmed_symptoms, symptom_df, n_diseases,
        mode=settings.INQUIRY_SCORING, top_k=top_k,
    )

    candidates = [
        CandidateDisease(
            name=s.name,
            base_confidence=s.base_confidence,
            confidence=s.base_confidence,   # 初始值，后续 apply_context_weights 调整
            matched_symptoms=s.matched_symptoms,
            all_symptoms=s.all_symptoms,    # 打分查询已带回，enrich 不再重复查
            symptom_weights=s.symptom_weights,
            department="",
            checks=[],
            complications=[],
        )
        for s in scored
    ]

    logger.debug(f"Neo4j 候选疾病: {[(c.name, c.base_confidence) for c in candidates]}")
    return candidates


async def enrich_candidate_details(
    candidates: list[CandidateDisease],
    neo4j_driver: AsyncDriver,
) -> list[CandidateDisease]:
    """
    补充候选疾病的完整信息：全部症状、建议科室、建议检查、并发症。
    在首次查到候选疾病后调用一次，后续轮次复用缓存在 state 里的数据。
    """
    if not candidates:
        return candidates

    disease_names = [c.name for c in candidates]

    # 查全部症状
    symptoms_cypher = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(s.name) AS symptoms
    """
    # 查科室
    dept_cypher = """
    MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department)
    WHERE d.name IN $names
    RETURN d.name AS disease, dept.name AS department
    """
    # 查检查项目
    check_cypher = """
    MATCH (d:Disease)-[:NEED_CHECK]->(c:Check)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(c.name) AS checks
    """
    # 查并发症
    comp_cypher = """
    MATCH (d:Disease)-[:ACOMPANY_WITH]->(comp:Disease)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(comp.name) AS complications
    """

    async with neo4j_driver.session() as session:
        symptoms_result = await (await session.run(symptoms_cypher, names=disease_names)).data()
        dept_result     = await (await session.run(dept_cypher,     names=disease_names)).data()
        check_result    = await (await session.run(check_cypher,    names=disease_names)).data()
        comp_result     = await (await session.run(comp_cypher,     names=disease_names)).data()

    # 构建查找字典
    symptoms_map    = {r["disease"]: r["symptoms"]      for r in symptoms_result}
    dept_map        = {r["disease"]: r["department"]    for r in dept_result}
    check_map       = {r["disease"]: r["checks"]        for r in check_result}
    comp_map        = {r["disease"]: r["complications"] for r in comp_result}

    for c in candidates:
        c.all_symptoms  = symptoms_map.get(c.name, [])
        c.department    = dept_map.get(c.name, "")
        c.checks        = check_map.get(c.name, [])
        c.complications = comp_map.get(c.name, [])

    return candidates


async def get_pending_symptoms(
    candidates: list[CandidateDisease],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    asked_symptoms: list[str],
) -> list[tuple[str, int]]:
    """
    从候选疾病的症状集合中，提取尚未确认/否认/追问过的症状。
    返回 [(症状名, 出现在几个候选疾病中)]，按出现次数升序排列。
    出现次数越少 = 区分度越高，优先追问。
    """
    already_known = set(confirmed_symptoms) | set(denied_symptoms) | set(asked_symptoms)
    symptom_count: dict[str, int] = {}
    symptom_weight: dict[str, float] = {}
    for c in candidates:
        for s in c.all_symptoms:
            if s not in already_known:
                symptom_count[s] = symptom_count.get(s, 0) + 1
                # 取该症状在各候选疾病中的最大归一化 IDF 权重（legacy 模式恒为 0）
                w = c.symptom_weights.get(s, 0.0)
                if w > symptom_weight.get(s, 0.0):
                    symptom_weight[s] = w

    # 主排序：出现在越少候选疾病中越优先（区分候选的能力强）
    # 次排序：IDF 权重越高越优先（确认/否认后对置信度影响大）
    sorted_symptoms = sorted(
        symptom_count.items(),
        key=lambda x: (x[1], -symptom_weight.get(x[0], 0.0)),
    )
    return sorted_symptoms
