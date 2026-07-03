"""
HyDE（Hypothetical Document Embeddings）模块。

职责：
1. LLM 根据用户问题生成假设性回答
2. 对假设回答做向量化，返回

返回的向量供 doc_rag.py 用于 Milvus 检索，比直接用原始问题向量召回率更高。
"""

from __future__ import annotations
from loguru import logger
from langchain_core.messages import SystemMessage
from src.agents.llm_utils import ainvoke_with_timeout
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.agents.knowledge.prompts import HYDE_PROMPT


async def generate_hyde_embedding(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
) -> list[float]:
    """
    HyDE（Hypothetical Document Embeddings）：
    1. LLM 生成假设性回答
    2. 对假设回答做向量化
    
    返回的向量供 doc_rag.py 用于 Milvus 检索，比直接用原始问题向量召回率更高。
    """
    prompt = HYDE_PROMPT.format(question=question)
    try:
        response = await ainvoke_with_timeout(llm, [SystemMessage(content=prompt)], step="knowledge.llm")
        hypothetical_doc = response.content.strip()
        logger.debug(f"HyDE 假设文档: {hypothetical_doc[:100]}...")
        return await embedding_model.aembed_query(hypothetical_doc)     # 返回假设回答的向量
    except Exception as e:
        logger.warning(f"HyDE 生成失败，回退到原始查询向量: {e}")
        return await embedding_model.aembed_query(question)