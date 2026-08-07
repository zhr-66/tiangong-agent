# src/agents/inquiry/confidence.py

from __future__ import annotations

from src.core.config import get_settings
from src.agents.inquiry.scoring import denial_penalty
from src.agents.inquiry.state import CandidateDisease, PatientContext


def apply_context_weights(
    candidates: list[CandidateDisease],
    patient_context: PatientContext,
    denied_symptoms: list[str],
) -> list[CandidateDisease]:
    """
    在基础置信度上叠加上下文权重，返回重新排序后的候选疾病列表。

    权重规则：
      +0.15  用户有该疾病的既往病史（PostgreSQL medical_history）
      +0.10  长期记忆中有相关记录（Milvus long_term_memories）
      +0.05  用户年龄/性别与该疾病易感人群匹配（Patient 表 easy_get 字段，预留）
      否认惩罚（scoring.denial_penalty，按打分模式分级）：
        idf_f1 : -β · Σ被否认症状的归一化 IDF 权重（否认特征症状重罚，可叠加）
        legacy : 独有核心症状被否认时一次性 -0.20（旧行为）
    """
    settings = get_settings()

    # 计算每个症状出现在几个候选疾病中（legacy 否认规则判断"核心症状"用）
    symptom_disease_count: dict[str, int] = {}
    for c in candidates:
        for s in c.all_symptoms:
            symptom_disease_count[s] = symptom_disease_count.get(s, 0) + 1

    for c in candidates:
        score = c.base_confidence

        # +0.15 既往病史命中
        for history_item in patient_context.medical_history:
            if c.name in history_item:
                score += 0.15
                break

        # +0.10 长期记忆命中
        for memory in patient_context.long_term_memories:
            if c.name in memory:
                score += 0.10
                break

        # +0.05 易感人群匹配（简化实现：暂用年龄段粗匹配，后续可接 easy_get 字段）
        # 此处预留接口，实际匹配逻辑需结合 Disease.easy_get 字段内容
        # score += 0.05  # 暂时注释，等 easy_get 字段接入后启用

        # 否认惩罚（idf_f1 模式按症状权重分级；legacy 模式保持旧规则）
        score += denial_penalty(
            denied_symptoms,
            c.all_symptoms,
            c.symptom_weights,
            symptom_disease_count,
            beta=settings.INQUIRY_DENIAL_BETA,
        )

        # 置信度限制在 [0, 1]
        c.confidence = round(max(0.0, min(1.0, score)), 4)

    # 按最终置信度降序排列
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates


def check_convergence(
    candidates: list[CandidateDisease],
    current_round: int,
    max_rounds: int = 10,
    top1_threshold: float | None = None,
    gap_threshold: float | None = None,
) -> tuple[bool, bool]:
    """
    判断问诊是否可以收敛（输出结论）。

    阈值默认读配置（INQUIRY_TOP1_THRESHOLD / INQUIRY_GAP_THRESHOLD），
    随打分模式校准——两种模式的分数分布不同，阈值不可混用。
    离线校准方式见 scripts/eval_confidence.py。

    Returns:
        (should_conclude, force_conclude)
        should_conclude : True = 可以输出结论
        force_conclude  : True = 是因为达到轮次上限被迫结束（需在结论中标注"信息不足"）
    """
    if top1_threshold is None or gap_threshold is None:
        settings = get_settings()
        top1_threshold = top1_threshold if top1_threshold is not None else settings.INQUIRY_TOP1_THRESHOLD
        gap_threshold = gap_threshold if gap_threshold is not None else settings.INQUIRY_GAP_THRESHOLD

    # 达到轮次上限，强制结束
    if current_round >= max_rounds:
        return True, True

    if not candidates:
        return False, False

    top1 = candidates[0].confidence

    # 条件1：Top1 置信度达到阈值
    if top1 >= top1_threshold:
        return True, False

    # 条件2：Top1 与 Top2 差值达到阈值（Top1 明显领先）
    if len(candidates) >= 2:
        top2 = candidates[1].confidence
        if top1 - top2 >= gap_threshold:
            return True, False

    return False, False
