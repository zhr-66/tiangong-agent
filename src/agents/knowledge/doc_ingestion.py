from __future__ import annotations
import os
import hashlib
from loguru import logger
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.knowledge.doc_rag import COLLECTION_NAME

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _create_collection_if_not_exists(milvus_client: MilvusClient) -> None:
    if milvus_client.has_collection(collection_name=COLLECTION_NAME):
        return
    
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=1024,
        metric_type="COSINE",
        index_params={
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
            "metric_type": "COSINE",
        },
        schema=[
            {"name": "id", "type": "VARCHAR", "max_length": 64, "is_primary": True},
            {"name": "doc_id", "type": "VARCHAR", "max_length": 32},
            {"name": "doc_name", "type": "VARCHAR", "max_length": 256},
            {"name": "doc_type", "type": "VARCHAR", "max_length": 32},
            {"name": "category", "type": "VARCHAR", "max_length": 32},
            {"name": "page_number", "type": "INT"},
            {"name": "chunk_index", "type": "INT"},
            {"name": "text", "type": "VARCHAR", "max_length": 65535},
            {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 1024},
        ],
    )
    logger.info(f"创建 collection: {COLLECTION_NAME}")


def ensure_knowledge_collection(milvus_client: MilvusClient) -> None:
    _create_collection_if_not_exists(milvus_client)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    text = text.replace("\n", " ").replace("\r", " ")
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind(".")
            last_comma = chunk.rfind("，")
            last_sep = max(last_period, last_comma)
            if last_sep > chunk_size // 2:
                chunk = chunk[:last_sep + 1]
                end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks


async def _process_file(file_path: str) -> list[tuple[str, int]]:
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        chunks = [(c, 1) for c in _chunk_text(content)]
    
    elif ext in (".pdf", ".docx", ".doc"):
        try:
            from llama_index.core import SimpleDirectoryReader
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            for doc in docs:
                page_num = doc.metadata.get("page_label", 1)
                try:
                    page_num = int(page_num)
                except:
                    page_num = 1
                text_chunks = _chunk_text(doc.text)
                for chunk in text_chunks:
                    chunks.append((chunk, page_num))
        except ImportError:
            logger.warning("llama_index 未安装，无法解析 PDF/Word")
            with open(file_path, "rb") as f:
                content = str(f.read()[:10000], errors="replace")
            chunks = [(c, 1) for c in _chunk_text(content)]
    
    elif ext == ".md":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        chunks = [(c, 1) for c in _chunk_text(content)]
    
    else:
        logger.warning(f"不支持的文件格式: {ext}")
    
    return chunks


async def ingest_file(
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> int:
    ensure_knowledge_collection(milvus_client)
    
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]
    
    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )
    
    chunks = await _process_file(file_path)
    if not chunks:
        logger.warning(f"文件 {doc_name} 解析后无内容")
        return 0
    
    all_data = []
    for i, (text, page_num) in enumerate(chunks):
        if not text.strip():
            continue
        
        emb = await embedding_model.aembed_query(text)
        data_item = {
            "id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_type": doc_type,
            "category": category,
            "page_number": page_num,
            "chunk_index": i,
            "text": text[:65000],
            "embedding": emb,
        }
        all_data.append(data_item)
    
    if all_data:
        for i in range(0, len(all_data), 1000):
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data[i:i + 1000],
            )
    
    logger.info(f"文档 {doc_name} 导入完成，共 {len(all_data)} 个分块")
    return len(all_data)


async def ingest_drug_instructions(
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    db_session: AsyncSession,
) -> int:
    ensure_knowledge_collection(milvus_client)
    
    from sqlalchemy import text
    result = await db_session.execute(
        text("SELECT id, name, description, indications, contraindications, dosage FROM drug_details WHERE description IS NOT NULL"),
    )
    rows = result.fetchall()
    
    total = 0
    for row in rows:
        drug_id, name, description, indications, contraindications, dosage = row
        
        content = f"药品名称：{name}\n"
        if description:
            content += f"药品说明：{description}\n"
        if indications:
            content += f"适应症：{indications}\n"
        if contraindications:
            content += f"禁忌症：{contraindications}\n"
        if dosage:
            content += f"用法用量：{dosage}\n"
        
        doc_name = f"{name}_说明书.txt"
        doc_id = hashlib.md5(f"drug_{drug_id}".encode()).hexdigest()[:16]
        
        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
        )
        
        chunks = _chunk_text(content)
        all_data = []
        for i, text in enumerate(chunks):
            emb = await embedding_model.aembed_query(text)
            data_item = {
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": "drug_instruction",
                "category": "药剂科",
                "page_number": 1,
                "chunk_index": i,
                "text": text[:65000],
                "embedding": emb,
            }
            all_data.append(data_item)
        
        if all_data:
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data,
            )
            total += len(all_data)
    
    logger.info(f"药品说明书导入完成，共 {total} 个分块")
    return total