from __future__ import annotations
import json
from loguru import logger

CONTEXT_TTL = 1800
MAX_HISTORY = 10


async def load_conversation_context(
    redis_client,
    user_id: str,
    session_id: str,
) -> list[dict]:
    """从 Redis 加载知识对话上下文。"""
    key = f"knowledge_ctx:{user_id}:{session_id}"
    try:
        raw = await redis_client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"加载知识对话上下文失败: {e}")
    return []


async def save_conversation_context(
    redis_client,
    user_id: str,
    session_id: str,
    history: list[dict],
) -> None:
    key = f"knowledge_ctx:{user_id}:{session_id}"
    trimmed = history[-MAX_HISTORY:]
    try:
        await redis_client.set(key, json.dumps(trimmed, ensure_ascii=False), ex=CONTEXT_TTL)
    except Exception as e:
        logger.warning(f"保存知识对话上下文失败: {e}")


async def append_turn(
    redis_client,
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
) -> list[dict]:
    """追加一轮对话到上下文，返回更新后的历史。"""
    history = await load_conversation_context(redis_client, user_id, session_id)
    history.append({
        "question": question,
        "answer": answer,
    })
    await save_conversation_context(redis_client, user_id, session_id, history)
    return history


def format_context(history: list[dict]) -> str:
    if not history:
        return ""
    parts = []
    for i, turn in enumerate(history, 1):
        parts.append(f"历史对话{i}：\n问：{turn['question']}\n答：{turn['answer']}")
    return "\n\n".join(parts)