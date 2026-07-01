from __future__ import annotations

from .prompts import (
    QUERY_REWRITE_PROMPT, HYDE_PROMPT, DOC_QA_PROMPT,
    ENTITY_EXTRACT_PROMPT, NL2CYPHER_PROMPT, GRAPH_QA_PROMPT,
    NL2SQL_PROMPT, SQL_QA_PROMPT, FUSION_PROMPT,
    HALLUCINATION_CHECK_PROMPT, PRESCRIPTION_PARSE_PROMPT,
    PRESCRIPTION_REPORT_PROMPT, ROUTE_PROMPT,
)
from .query_rewriter import rewrite_query
from .hyde import generate_hyde_embedding
from .reranker import rerank_docs
from .doc_rag import search_docs, search_docs_raw, format_doc_context, COLLECTION_NAME
from .graph_rag import search_graph, search_graph_raw
from .nl2sql import search_sql, search_sql_raw
from .hallucination_check import check_hallucination
from .fusion import multi_channel_search
from .prescription_review import review_prescription
from .audit import QueryAuditLog, Timer
from .feedback import save_feedback, get_feedback_stats
from .notification import notify_doc_update, get_unread_notifications, mark_notifications_read
from .conversation import load_conversation_context, save_conversation_context, append_turn, format_context
from .doc_ingestion import ingest_file, ingest_drug_instructions, ensure_knowledge_collection

__all__ = [
    "QUERY_REWRITE_PROMPT", "HYDE_PROMPT", "DOC_QA_PROMPT",
    "ENTITY_EXTRACT_PROMPT", "NL2CYPHER_PROMPT", "GRAPH_QA_PROMPT",
    "NL2SQL_PROMPT", "SQL_QA_PROMPT", "FUSION_PROMPT",
    "HALLUCINATION_CHECK_PROMPT", "PRESCRIPTION_PARSE_PROMPT",
    "PRESCRIPTION_REPORT_PROMPT", "ROUTE_PROMPT",
    "rewrite_query", "generate_hyde_embedding", "rerank_docs",
    "search_docs", "search_docs_raw", "format_doc_context", "COLLECTION_NAME",
    "search_graph", "search_graph_raw",
    "search_sql", "search_sql_raw",
    "check_hallucination",
    "multi_channel_search",
    "review_prescription",
    "QueryAuditLog", "Timer",
    "save_feedback", "get_feedback_stats",
    "notify_doc_update", "get_unread_notifications", "mark_notifications_read",
    "load_conversation_context", "save_conversation_context", "append_turn", "format_context",
    "ingest_file", "ingest_drug_instructions", "ensure_knowledge_collection",
]