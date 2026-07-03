from __future__ import annotations

import re


ROUTE_DOC_RAG = "doc_rag"
ROUTE_GRAPH_RAG = "graph_rag"
ROUTE_NL2SQL = "nl2sql"
ROUTE_MULTI = "multi"
ROUTE_PRESCRIPTION = "prescription"

VALID_KNOWLEDGE_ROUTES = {
    ROUTE_DOC_RAG,
    ROUTE_GRAPH_RAG,
    ROUTE_NL2SQL,
    ROUTE_MULTI,
    ROUTE_PRESCRIPTION,
}

PERSONAL_SYMPTOM_HINTS = (
    "我头疼", "我头痛", "我发烧", "我发热", "我咳嗽", "我胸闷", "我胸痛",
    "我肚子疼", "我腹痛", "我不舒服", "我难受", "我恶心", "我呕吐",
    "我腹泻", "本人头疼", "本人头痛", "最近头疼", "最近头痛", "最近发烧",
    "最近发热", "这几天", "这两天", "昨晚", "今天早上",
)

MEDICAL_SUBJECT_HINTS = (
    "病", "炎", "癌", "综合征", "高血压", "糖尿病", "感冒", "头痛",
    "头疼", "腹痛", "咳嗽", "发热", "发烧", "药", "片", "胶囊",
    "颗粒", "注射液", "阿司匹林", "布洛芬", "阿莫西林", "氨氯地平",
    "二甲双胍", "处方", "说明书", "指南", "检查", "症状", "病因",
    "禁忌", "适应症", "诊断", "治疗", "预防",
)

KNOWLEDGE_QUESTION_HINTS = (
    "是什么", "有哪些", "有什么", "表现", "症状", "病因", "原因",
    "怎么治疗", "治疗方案", "治疗方式", "能治好吗", "注意什么",
    "注意事项", "早期", "并发症", "区别", "诊断标准", "预防",
    "吃什么药", "用什么药", "常用药", "推荐药", "适应症", "禁忌",
    "不良反应", "用法用量", "指南", "说明书", "能一起吃", "同服",
    "配伍", "相互作用", "处方审核", "库存", "排名", "统计",
)

PRESCRIPTION_HINTS = (
    "处方审核", "审核处方", "处方", "配伍", "配伍禁忌", "相互作用",
    "一起吃", "一起服用", "同服", "联用", "能不能一起", "可以一起",
    "用药安全吗", "用药安全", "禁忌",
)

SQL_HINTS = (
    "统计", "数量", "多少例", "多少人", "排名", "排行", "趋势", "库存",
    "问诊量", "就诊量", "科室", "上个月", "本月", "本周", "同比", "环比",
    "Top", "top",
)

DOC_HINTS = (
    "说明书", "指南", "规范", "制度", "流程", "适应症", "用法用量",
    "不良反应", "注意事项", "病因", "原因", "预防", "治疗方式",
    "治疗方案", "诊断标准", "描述", "是什么",
)

GRAPH_HINTS = (
    "常用药", "推荐药", "吃什么药", "用什么药", "症状", "早期表现",
    "表现", "检查", "挂什么科", "看什么科", "并发症", "可能是什么病",
    "可能疾病", "属于什么科", "需要做什么检查",
)

MULTI_HINTS = (
    "合并", "伴有", "同时患", "同时有", "并且", "并发",
)

COMBINER_PATTERN = re.compile(r"(和|及|以及|同时|并且|还有|，|,|、)")


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _has_personal_symptom(text: str) -> bool:
    if _contains_any(text, PERSONAL_SYMPTOM_HINTS):
        return True
    return bool(re.search(r"(我|本人|家里人|孩子|老人).{0,8}(疼|痛|发烧|发热|咳嗽|难受|不舒服|恶心|呕吐|腹泻)", text))


def looks_like_knowledge_question(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if _has_personal_symptom(text):
        return False
    return _contains_any(text, MEDICAL_SUBJECT_HINTS) and _contains_any(text, KNOWLEDGE_QUESTION_HINTS)


def infer_knowledge_route(question: str) -> str | None:
    text = (question or "").strip()
    if not text:
        return None

    if _contains_any(text, PRESCRIPTION_HINTS):
        return ROUTE_PRESCRIPTION

    if _contains_any(text, SQL_HINTS):
        return ROUTE_NL2SQL

    has_doc = _contains_any(text, DOC_HINTS)
    has_graph = _contains_any(text, GRAPH_HINTS)

    if _contains_any(text, MULTI_HINTS) and _contains_any(text, ("药", "治疗", "用药")):
        return ROUTE_MULTI

    if has_doc and has_graph and COMBINER_PATTERN.search(text):
        return ROUTE_MULTI

    if has_graph:
        return ROUTE_GRAPH_RAG

    if has_doc:
        return ROUTE_DOC_RAG

    return None


def normalize_knowledge_route(route: str | None) -> str | None:
    if route in VALID_KNOWLEDGE_ROUTES:
        return route
    return None
