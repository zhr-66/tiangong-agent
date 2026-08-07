# src/api/routers/chat.py

from __future__ import annotations
import asyncio
import json
import traceback
from typing import Awaitable, Callable
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.infra.database import get_db
from src.infra.redis_cache import get_checkpointer_redis
from src.agents.llm_utils import reset_token_sink, set_token_sink
from src.agents.supervisor_agent import get_supervisor_agent, UserContext
from src.agents.inquiry.graph import run_inquiry, build_inquiry_deps
from src.agents.inquiry.state import InquiryState, InquiryPhase
from src.agents.workers.inquiry_agent import handle_handoff
from src.agents.workers.knowledge_agent import get_knowledge_agent

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    patient_id: int | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str

INTERNAL_REPLY_MARKERS = (
    "## SESSION INTENT",
    "## SUMMARY",
    "## ARTIFACTS",
    "## NEXT STEPS",
    "SESSION INTENT",
    "Here is a summary of the conversation to date:",
)

SAFE_INQUIRY_FALLBACK = (
    "请描述一下具体症状，包括不舒服的部位、持续多久、严重程度，"
    "以及是否伴随发热、疼痛、呕吐、胸闷或呼吸困难等情况。"
)



# \u2500\u2500 API \u5c42\u5feb\u901f\u5206\u6d41 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# \u539f\u5219\uff1a\u5feb\u901f\u901a\u9053\u53ea\u5904\u7406"\u9ad8\u7f6e\u4fe1\u72ec\u5360\u547d\u4e2d"\uff0c\u4e3a\u5e38\u89c1\u8bf7\u6c42\u7701\u4e00\u8df3 Supervisor LLM \u8c03\u7528\uff1b
# \u540c\u65f6\u547d\u4e2d\u4e24\u7c7b\u6216\u90fd\u4e0d\u547d\u4e2d\u7684\u6b67\u4e49\u6d88\u606f\uff0c\u4e00\u5f8b\u4ea4 Supervisor \u7531 LLM \u7ed3\u5408\u4e0a\u4e0b\u6587\u8def\u7531\u3002
# \u5173\u952e\u8bcd\u6309"\u7cbe\u786e\u7387\u4f18\u5148"\u6311\u9009\u2014\u2014\u5feb\u901f\u901a\u9053\u7701\u4e0b\u7684\u5ef6\u8fdf\u53ea\u6709\u5728\u4e0d\u51fa\u9519\u65f6\u624d\u6709\u4ef7\u503c\u3002

# \u95ee\u8bca\u4fe1\u53f7\uff1a\u75c7\u72b6\u63cf\u8ff0\u3001\u5c31\u8bca/\u79d1\u5ba4\u5f15\u5bfc
INQUIRY_HINTS = (
    "\u5934\u75bc", "\u5934\u75db", "\u53d1\u70e7", "\u53d1\u70ed", "\u54b3\u55fd",
    "\u80f8\u95f7", "\u80f8\u75db", "\u547c\u5438\u56f0\u96be", "\u809a\u5b50\u75bc",
    "\u8179\u75db", "\u4e0d\u8212\u670d", "\u96be\u53d7", "\u6076\u5fc3", "\u5455\u5410",
    "\u8179\u6cfb", "\u6302\u4ec0\u4e48\u79d1", "\u5e94\u8be5\u6302", "\u770b\u4ec0\u4e48\u79d1",
    "\u5c31\u8bca",
)

# \u77e5\u8bc6\u95ee\u7b54\u4fe1\u53f7\uff1a\u75be\u75c5/\u836f\u54c1\u77e5\u8bc6\u7684\u7591\u95ee\u53e5\u5f0f
KNOWLEDGE_HINTS = (
    "\u662f\u4ec0\u4e48", "\u6709\u54ea\u4e9b", "\u7c7b\u578b", "\u539f\u56e0",
    "\u600e\u4e48\u6cbb\u7597", "\u5982\u4f55\u6cbb\u7597", "\u80fd\u6cbb\u597d\u5417",
    "\u6ce8\u610f\u4ec0\u4e48", "\u65e9\u671f\u75c7\u72b6", "\u5e76\u53d1\u75c7",
    "\u533a\u522b", "\u8bca\u65ad\u6807\u51c6", "\u9884\u9632",
    "\u5403\u4ec0\u4e48\u836f", "\u5403\u54ea\u79cd\u836f", "\u7528\u4ec0\u4e48\u836f",
    "\u5e38\u7528\u836f", "\u63a8\u8350\u836f", "\u5bf9\u4ec0\u4e48\u836f",
)


def decide_fast_route(message: str) -> tuple[str, list[str]]:
    """
    API \u5c42\u5feb\u901f\u5206\u6d41\uff08\u7eaf\u51fd\u6570\uff0c\u4fbf\u4e8e\u79bb\u7ebf\u56de\u5f52\u6d4b\u8bd5\uff09\u3002

    \u8fd4\u56de (\u8def\u7531, \u547d\u4e2d\u8bcd\u5217\u8868)\uff1a
      "knowledge"  \u53ea\u547d\u4e2d\u77e5\u8bc6\u5173\u952e\u8bcd \u2192 \u76f4\u8fde\u77e5\u8bc6 Agent
      "inquiry"    \u53ea\u547d\u4e2d\u95ee\u8bca\u5173\u952e\u8bcd \u2192 \u76f4\u8fde\u95ee\u8bca\u56fe
      "supervisor" \u540c\u65f6\u547d\u4e2d\u6216\u90fd\u4e0d\u547d\u4e2d \u2192 \u4ea4 Supervisor LLM \u8def\u7531
    """
    inquiry_hits = [h for h in INQUIRY_HINTS if h in message]
    knowledge_hits = [h for h in KNOWLEDGE_HINTS if h in message]

    if knowledge_hits and not inquiry_hits:
        return "knowledge", knowledge_hits
    if inquiry_hits and not knowledge_hits:
        return "inquiry", inquiry_hits
    return "supervisor", inquiry_hits + knowledge_hits

def _content_to_text(content) -> str:
    """Normalize LangChain message content to displayable plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _has_internal_markers(text: str) -> bool:
    return any(marker in text for marker in INTERNAL_REPLY_MARKERS)


def _is_user_facing_candidate(block: str) -> bool:
    stripped = block.strip()
    if not stripped or _has_internal_markers(stripped):
        return False

    internal_prefixes = (
        "用户",
        "助手",
        "目前",
        "此前",
        "None",
        "1.",
        "2.",
        "3.",
        "- ",
        "##",
        "Here is",
    )
    if stripped.startswith(internal_prefixes):
        return False

    forbidden_terms = (
        "patient_id",
        "call_inquiry_agent",
        "工具调用",
        "参数",
        "API",
        "Redis",
        "SESSION",
        "ARTIFACTS",
        "NEXT STEPS",
    )
    return not any(term in stripped for term in forbidden_terms)


def sanitize_user_reply(text: str) -> str:
    """
    Remove checkpoint/summarization/meta text that must never be shown to users.

    Some LangGraph/LangChain middleware can add conversation summaries into the
    message stream. If those summaries accidentally become assistant content,
    keep only the last normal user-facing paragraph.
    """
    normalized = (text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return SAFE_INQUIRY_FALLBACK

    if not _has_internal_markers(normalized):
        return normalized

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    for paragraph in reversed(paragraphs):
        if _is_user_facing_candidate(paragraph):
            return paragraph

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for line in reversed(lines):
        if _is_user_facing_candidate(line):
            return line

    return SAFE_INQUIRY_FALLBACK


def extract_agent_reply(result: dict) -> str:
    """Pick the last real assistant message and sanitize it for UI display."""
    messages = result.get("messages") or []
    fallback_text = ""

    for msg in reversed(messages):
        content = _content_to_text(getattr(msg, "content", ""))
        if not content.strip():
            continue
        if isinstance(msg, AIMessage):
            cleaned = sanitize_user_reply(content)
            if cleaned:
                return cleaned
        fallback_text = content

    return sanitize_user_reply(fallback_text)


def _make_keys(user_id: str, session_id: str) -> tuple[str, str]:
    """生成 Redis 键名。thread_id 与 Supervisor checkpointer 保持一致。"""
    thread_id = f"{user_id}:{session_id}"
    return f"inquiry_active:{thread_id}", f"inquiry_state:{thread_id}"


async def _run_inquiry_turn(
    message: str,
    thread_id: str,
    redis,
    db,
) -> str:
    """
    执行一轮问诊对话（路由层直接调用，绕过 Supervisor）。
    从 Redis 恢复状态 → 执行 InquiryGraph → 保存新状态 → 返回回复。
    """
    active_key = f"inquiry_active:{thread_id}"
    state_key  = f"inquiry_state:{thread_id}"

    # 从 Redis 反序列化恢复上一轮状态
    raw = await redis.get(state_key)
    existing_state = InquiryState.model_validate_json(raw) if raw else None

    deps = build_inquiry_deps(db_session=db)
    reply, new_state = await run_inquiry(
        user_message=message,
        thread_id=thread_id,
        deps=deps,
        existing_state=existing_state,
    )

    # 问诊结束：清除 Redis 标记，触发挂号移交
    if new_state.phase in (InquiryPhase.HANDOFF, InquiryPhase.END):
        await redis.delete(active_key, state_key)
        if new_state.phase == InquiryPhase.HANDOFF and new_state.handoff_payload:
            handoff_reply = await handle_handoff(new_state.handoff_payload)
            return sanitize_user_reply(f"{reply}\n\n---\n{handoff_reply}")
        return reply

    # 问诊继续：更新 Redis 状态，重置 TTL
    await redis.set(state_key,  new_state.model_dump_json(), ex=3600)
    await redis.set(active_key, "1",                         ex=3600)
    return reply


# ── 非流式接口 ────────────────────────────────────────────────────────────
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    非流式对话接口。
    路由层判断是否有活跃问诊：有则直接走 InquiryGraph，无则走 Supervisor。
    """
    try:
        redis = get_checkpointer_redis()
        thread_id = f"{req.user_id}:{req.session_id}"
        active_key = f"inquiry_active:{thread_id}"

        # ── 问诊进行中：直接走 InquiryGraph ──
        if await redis.exists(active_key):
            reply = await _run_inquiry_turn(req.message, thread_id, redis, db)
            return ChatResponse(reply=reply, session_id=req.session_id)

        # ── 无活跃问诊：快速分流（独占命中直连；歧义/未命中走 Supervisor） ──
        route, hits = decide_fast_route(req.message)
        logger.info("chat 快速分流: route={} hits={} message={!r}", route, hits, req.message[:50])

        if route == "knowledge":
            knowledge_agent = await get_knowledge_agent()
            reply = await knowledge_agent.query(
                req.message,
                user_id=req.user_id,
                session_id=req.session_id,
                db_session=db,
            )
            return ChatResponse(reply=reply, session_id=req.session_id)

        if route == "inquiry":
            reply = await _run_inquiry_turn(req.message, thread_id, redis, db)
            return ChatResponse(reply=reply, session_id=req.session_id)

        agent = await get_supervisor_agent()
        config = {"configurable": {"thread_id": thread_id}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            context=UserContext(user_id=req.user_id, session_id=req.session_id),
        )
        reply = extract_agent_reply(result)
        return ChatResponse(reply=reply, session_id=req.session_id)

    except Exception as e:
        logger.exception(f"chat 接口异常")
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── 流式接口（SSE） ───────────────────────────────────────────────────────
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_pipeline_events(
    pipeline: Callable[[], Awaitable[str]],
    session_id: str,
):
    """
    把回答管线包装成 SSE 事件流（真流式）。

    机制：注册 token sink（contextvar，自动传播到管线内部的 LangGraph 节点/工具），
    管线中"面向用户的最终生成"（agenerate_final 调用点）逐 token 实时推送；
    管线结束后推一条 replace 事件对账——权威回复可能与已推 token 存在差异
    （幻觉检测追加的警告、DocRAG 回退 GraphRAG、挂号移交拼接等生成后修正），
    客户端以 replace 内容为准整体替换。
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def sink(token: str) -> None:
        await queue.put(token)

    async def runner() -> str:
        ctx_token = set_token_sink(sink)
        try:
            return await pipeline()
        finally:
            reset_token_sink(ctx_token)
            await queue.put(None)  # 结束哨兵，确保消费循环退出

    task = asyncio.create_task(runner())
    try:
        while True:
            token = await queue.get()
            if token is None:
                break
            yield _sse({"type": "token", "content": token})

        reply = await task  # 管线抛异常时在此重新抛出，由外层统一转 error 事件
        yield _sse({"type": "replace", "content": reply})
        yield _sse({"type": "done", "session_id": session_id})
    finally:
        if not task.done():
            task.cancel()  # 客户端断开时终止管线


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    流式对话接口（Server-Sent Events）。
    各分支的"最终面向用户生成"逐 token 实时推送。

    客户端接收格式：
        data: {"type": "token",   "content": "..."}   # 增量 token，追加显示
        data: {"type": "replace", "content": "..."}   # 权威全文，整体替换已显示内容
        data: {"type": "done",    "session_id": "..."}
        data: {"type": "error",   "message": "..."}
    """
    async def event_generator():
        try:
            redis = get_checkpointer_redis()
            thread_id = f"{req.user_id}:{req.session_id}"
            active_key = f"inquiry_active:{thread_id}"

            # 分支选择逻辑与非流式接口一致
            route = "inquiry_active" if await redis.exists(active_key) else None
            if route is None:
                route, hits = decide_fast_route(req.message)
                logger.info(
                    "chat/stream 快速分流: route={} hits={} message={!r}",
                    route, hits, req.message[:50],
                )

            if route == "inquiry_active" or route == "inquiry":
                def pipeline():
                    return _run_inquiry_turn(req.message, thread_id, redis, db)

            elif route == "knowledge":
                knowledge_agent = await get_knowledge_agent()

                def pipeline():
                    return knowledge_agent.query(
                        req.message,
                        user_id=req.user_id,
                        session_id=req.session_id,
                        db_session=db,
                    )

            else:
                # Supervisor 分支：Supervisor 自身的最终回复不逐 token 推送
                # （经 extract_agent_reply 清洗后随 replace 事件下发），
                # 但其调用的知识类工具内部的最终生成仍会实时流出。
                async def pipeline():
                    agent = await get_supervisor_agent()
                    config = {"configurable": {"thread_id": thread_id}}
                    result = await agent.ainvoke(
                        {"messages": [{"role": "user", "content": req.message}]},
                        config=config,
                        context=UserContext(user_id=req.user_id, session_id=req.session_id),
                    )
                    return extract_agent_reply(result)

            async for event in _stream_pipeline_events(pipeline, req.session_id):
                yield event

        except Exception:
            logger.exception("chat/stream 接口异常")
            yield _sse({"type": "error", "message": traceback.format_exc()})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



