from __future__ import annotations
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from loguru import logger
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.infra.database import get_db
from src.agents.knowledge import (
    ingest_file, ensure_knowledge_collection,
    save_feedback, get_feedback_stats,
    notify_doc_update, get_unread_notifications, mark_notifications_read,
)
from src.agents.knowledge.doc_rag import COLLECTION_NAME

settings = get_settings()
router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def _get_deps():
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return embedding_model, milvus_client


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(..., description="文档类型：guideline/drug_instruction/sop/literature"),
    category: str = Form("通用", description="所属分类：内科/外科/药剂科/行政等"),
):
    allowed_ext = {".pdf", ".docx", ".doc", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"不支持的文件格式: {ext}，支持: {allowed_ext}")

    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        embedding_model, milvus_client = _get_deps()
        chunk_count = await ingest_file(
            file_path=tmp_path,
            doc_name=file.filename,
            doc_type=doc_type,
            category=category,
            embedding_model=embedding_model,
            milvus_client=milvus_client,
        )
        return {"message": f"文档 '{file.filename}' 导入成功", "chunks": chunk_count}
    finally:
        os.unlink(tmp_path)


@router.post("/upload/notify")
async def upload_document_with_notify(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    category: str = Form("通用"),
    db: AsyncSession = Depends(get_db),
):
    result = await upload_document(file=file, doc_type=doc_type, category=category)

    try:
        await notify_doc_update(db, file.filename, doc_type, category, action="upload")
    except Exception as e:
        logger.warning(f"发送知识更新通知失败: {e}")

    return result


@router.delete("/docs/{doc_name}")
async def delete_document(doc_name: str):
    import hashlib
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]
    _, milvus_client = _get_deps()
    try:
        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_name}"',
        )
        return {"message": f"文档 '{doc_name}' 已删除"}
    except Exception as e:
        raise HTTPException(500, f"删除失败: {e}")


@router.get("/docs")
async def list_documents():
    _, milvus_client = _get_deps()
    try:
        ensure_knowledge_collection(milvus_client)
        results = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter="chunk_index == 0",
            output_fields=["doc_name", "doc_type", "category"],
            limit=500,
        )
        docs = [
            {"doc_name": r["doc_name"], "doc_type": r["doc_type"], "category": r["category"]}
            for r in results
        ]
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(500, f"查询失败: {e}")


class FeedbackRequest(BaseModel):
    user_id: str
    question: str
    answer: str
    rating: int
    comment: str = ""
    intent: str = ""
    channels: str = ""


@router.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    feedback_id = await save_feedback(
        db=db, user_id=req.user_id, question=req.question,
        answer=req.answer, rating=req.rating, comment=req.comment,
        intent=req.intent, channels=req.channels,
    )
    return {"feedback_id": feedback_id, "message": "反馈已记录"}


@router.get("/feedback/stats")
async def feedback_stats(db: AsyncSession = Depends(get_db)):
    return await get_feedback_stats(db)


@router.get("/notifications")
async def list_notifications(
    category: str = None,
    db: AsyncSession = Depends(get_db),
):
    notifications = await get_unread_notifications(db, category=category)
    return {"notifications": notifications, "total": len(notifications)}


class MarkReadRequest(BaseModel):
    ids: list[int]


@router.post("/notifications/read")
async def mark_read(
    req: MarkReadRequest,
    db: AsyncSession = Depends(get_db),
):
    await mark_notifications_read(db, req.ids)
    return {"message": f"已标记 {len(req.ids)} 条通知为已读"}