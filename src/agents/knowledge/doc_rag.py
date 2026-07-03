"""
文档 RAG 模块。

职责：
1. HyDE 增强：生成假设文档向量，提升召回率
2. Milvus 向量粗检索：从知识库中召回 top_k 文档片段
3. Reranker 精排：对粗检索结果做二次排序
4. LLM 生成回答：基于精排后的文档片段生成自然语言回答

典型场景：查询文档内容，如"阿莫西林说明书怎么说"、"高血压诊疗指南有哪些推荐"
"""

from __future__ import annotations
from loguru import logger
from langchain_core.messages import SystemMessage
from src.agents.llm_utils import ainvoke_with_timeout
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient

from src.agents.knowledge.prompts import DOC_QA_PROMPT

COLLECTION_NAME = "knowledge_docs"


async def search_docs_raw(
    question: str,      # 用户原始问题
    embedding_model: Embeddings,    # 向量化模型
    milvus_client: MilvusClient,    
    top_k: int = 20,     # 向量初召回最多20条
    rerank_top_k: int = 5,   # Reranker精排后保留5条最优片段
    doc_type: str | None = None,    # 过滤文档类型（如指南/病历/教材）
    llm: BaseChatModel | None = None,   # HyDE需要的大模型
    use_hyde: bool = False,     # 是否开启HyDE假设文档增强
) -> list[dict]:
    if use_hyde and llm is not None:
        from src.agents.knowledge.hyde import generate_hyde_embedding      
        query_vec = await generate_hyde_embedding(question, llm, embedding_model)   # 生成假设回答的向量
    else:
        query_vec = await embedding_model.aembed_query(question)

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    filter_expr = f'doc_type == "{doc_type}"' if doc_type else None

    try:
        # 向量初召回
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vec],
            limit=top_k,
            output_fields=["doc_name", "doc_type", "page_number", "chunk_index", "text"],
            search_params=search_params,
            filter=filter_expr,
        )
    except Exception as e:
        logger.warning(f"文档检索失败: {e}")
        return []

    if not results or not results[0]:
        return []

    hits = [
        {**hit["entity"], "score": hit.get("distance", 0.0)}
        for hit in results[0]
    ]

    from src.agents.knowledge.reranker import rerank_docs
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)     # Reranker精排
    return reranked


def format_doc_context(hits: list[dict]) -> str:
    """将检索结果格式化为 LLM 可读的上下文字符串。
       统一格式: 片段1 [文档名, 第X页]: 内容... 。
    """
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        source = f"[{hit['doc_name']}, 第{hit.get('page_number', '?')}页]"
        parts.append(f"片段{i} {source}:\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


async def search_docs(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    role: str = "patient",
    use_hyde: bool = True,
    context_text: str = "",
) -> str:
    """HyDE 增强 + 文档 RAG 检索 + Reranker 精排 + 生成回答。
    question: 用于检索的问题（改写后的问题）
    context_text: 用于最终 LLM 生成回答的问题（含多轮对话上下文）
    """
    hits = await search_docs_raw(
        question, embedding_model, milvus_client,
        top_k=top_k, rerank_top_k=rerank_top_k, doc_type=doc_type,
        llm=llm, use_hyde=use_hyde,
    )
    if not hits:
        return "当前知识库中未找到与您问题相关的文档内容。"

    context = format_doc_context(hits)
    answer_question = context_text if context_text else question
    prompt = DOC_QA_PROMPT.format(question=answer_question, context=context, role=role)
    response = await ainvoke_with_timeout(llm, [SystemMessage(content=prompt)], step="knowledge.llm")
    return response.content