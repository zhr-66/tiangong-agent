"""
文档导入模块。

职责：
1. 解析多种格式文档（PDF、Word、TXT、Markdown）
2. 文本分块（按固定长度 + 语义边界切分）
3. 向量化后写入 Milvus knowledge_docs collection
4. 支持从 PostgreSQL 药品表批量导入药品说明书
"""

from __future__ import annotations
import asyncio
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

    from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    schema = CollectionSchema(fields=fields, description="知识文档向量库")

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )

    # 创建向量索引
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )
    milvus_client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )

    # 加载集合到内存
    milvus_client.load_collection(collection_name=COLLECTION_NAME)
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
        # 优先使用 MinerU 高精度解析（表格/公式还原）
        md_text = await _parse_with_mineru(file_path, os.path.basename(file_path))
        if md_text:
            chunks = [(c, 1) for c in _chunk_text(md_text)]
        else:
            # 回退到 LlamaIndex
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


async def _parse_with_mineru(file_path: str, file_name: str) -> str | None:
    """尝试用 MinerU 解析文档，失败则返回 None。"""
    try:
        from src.agents.knowledge.mineru_client import parse_document
        md_text = await parse_document(file_path, file_name)
        if md_text and len(md_text.strip()) > 10:
            logger.info(f"MinerU 解析成功: {file_name} ({len(md_text)} chars)")
            return md_text
    except Exception as e:
        logger.warning(f"MinerU 解析失败，回退到 LlamaIndex: {e}")
    return None


async def ingest_file(
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> int:
    # pymilvus 是同步客户端，统一放线程池执行，避免阻塞事件循环
    await asyncio.to_thread(ensure_knowledge_collection, milvus_client)

    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]

    await asyncio.to_thread(
        milvus_client.delete,
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
            await asyncio.to_thread(
                milvus_client.insert,
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
        text("""
            SELECT d.id, d.name, dd.indication, dd.usage_dosage,
                   dd.adverse_reaction, dd.contraindication, dd.precaution,
                   dd.interaction, dd.full_instruction
            FROM drugs d
            JOIN drug_details dd ON dd.drug_id = d.id
            WHERE dd.full_instruction IS NOT NULL
               OR dd.indication IS NOT NULL
        """),
    )
    rows = result.fetchall()

    total = 0
    for row in rows:
        drug_id, name, indication, usage_dosage, adverse_reaction, contraindication, precaution, interaction, full_instruction = row

        content = f"药品名称：{name}\n"
        if full_instruction:
            content += f"完整说明书：{full_instruction}\n"
        if indication:
            content += f"适应症：{indication}\n"
        if usage_dosage:
            content += f"用法用量：{usage_dosage}\n"
        if adverse_reaction:
            content += f"不良反应：{adverse_reaction}\n"
        if contraindication:
            content += f"禁忌：{contraindication}\n"
        if precaution:
            content += f"注意事项：{precaution}\n"
        if interaction:
            content += f"药物相互作用：{interaction}\n"

        doc_name = f"{name}_说明书.txt"
        doc_id = hashlib.md5(f"drug_{drug_id}".encode()).hexdigest()[:16]
        
        await asyncio.to_thread(
            milvus_client.delete,
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
            await asyncio.to_thread(
                milvus_client.insert,
                collection_name=COLLECTION_NAME,
                data=all_data,
            )
            total += len(all_data)
    
    logger.info(f"药品说明书导入完成，共 {total} 个分块")
    return total


async def ingest_diseases(
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    db_session: AsyncSession,
    batch_size: int = 100,
) -> int:
    """从 PostgreSQL diseases 表导入疾病知识到 Milvus。"""
    ensure_knowledge_collection(milvus_client)

    from sqlalchemy import text
    result = await db_session.execute(
        text("""
            SELECT id, name, description, cause, prevent, cure_way,
                   cured_prob, easy_get
            FROM diseases
            WHERE description IS NOT NULL
        """),
    )
    rows = result.fetchall()
    logger.info(f"从 diseases 表读取到 {len(rows)} 条疾病数据")

    total = 0
    all_data = []

    for row in rows:
        dis_id, name, description, cause, prevent, cure_way, cured_prob, easy_get = row

        content = f"疾病名称：{name}\n"
        if description:
            content += f"疾病描述：{description}\n"
        if cause:
            content += f"病因：{cause}\n"
        if prevent:
            content += f"预防：{prevent}\n"
        if cure_way:
            content += f"治疗方式：{cure_way}\n"
        if cured_prob:
            content += f"治愈率：{cured_prob}\n"
        if easy_get:
            content += f"易感人群：{easy_get}\n"

        doc_name = f"{name}_疾病知识.txt"
        doc_id = hashlib.md5(f"disease_{dis_id}".encode()).hexdigest()[:16]

        chunks = _chunk_text(content)
        for i, chunk_text in enumerate(chunks):
            emb = await embedding_model.aembed_query(chunk_text)
            all_data.append({
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": "disease_knowledge",
                "category": "内科",
                "page_number": 1,
                "chunk_index": i,
                "text": chunk_text[:65000],
                "embedding": emb,
            })
            total += 1

        # 批量写入
        if len(all_data) >= batch_size:
            await asyncio.to_thread(
                milvus_client.insert, collection_name=COLLECTION_NAME, data=all_data
            )
            logger.info(f"已导入 {total} 个分块...")
            all_data = []

    if all_data:
        await asyncio.to_thread(
            milvus_client.insert, collection_name=COLLECTION_NAME, data=all_data
        )

    logger.info(f"疾病知识导入完成，共 {total} 个分块")
    return total