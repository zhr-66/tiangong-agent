from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage
from loguru import logger


DEFAULT_LLM_TIMEOUT_SECONDS = 45

# 当前请求的 token 回调（流式 SSE 接口注册）。
# 注册后，agenerate_final 会逐 token 生成并实时回调；未注册时行为与 ainvoke_with_timeout 一致。
# 用 contextvar 隔离并发请求，且能自动传播到 LangGraph 节点等子任务中。
_token_sink: contextvars.ContextVar[Callable[[str], Awaitable[None]] | None] = (
    contextvars.ContextVar("llm_token_sink", default=None)
)


def set_token_sink(sink: Callable[[str], Awaitable[None]]) -> contextvars.Token:
    """注册当前上下文的 token 回调，返回用于 reset 的 token。"""
    return _token_sink.set(sink)


def reset_token_sink(token: contextvars.Token) -> None:
    _token_sink.reset(token)


async def ainvoke_with_timeout(
    llm: Any,
    messages: list[Any],
    *,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    step: str = "llm",
) -> Any:
    """Run an async LLM call with a hard upper bound."""
    try:
        return await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("{} timed out after {}s", step, timeout_seconds)
        raise


async def agenerate_final(
    llm: Any,
    messages: list[Any],
    *,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    step: str = "llm",
) -> Any:
    """面向用户的最终回答生成。

    仅用于"生成结果会直接展示给用户"的调用点（各 RAG 通道的最终生成、
    问诊的澄清/追问/结论等）。中间步骤（改写、路由、JSON 解析类）仍用
    ainvoke_with_timeout。

    流式 SSE 接口注册了 token sink 时：逐 token 生成并实时推送；
    未注册（非流式接口、脚本、评估）时：等价于 ainvoke_with_timeout。
    """
    sink = _token_sink.get()
    if sink is None:
        return await ainvoke_with_timeout(
            llm, messages, timeout_seconds=timeout_seconds, step=step
        )

    async def _stream_once() -> AIMessage:
        parts: list[str] = []
        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                await sink(text)
        return AIMessage(content="".join(parts))

    try:
        return await asyncio.wait_for(_stream_once(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("{} (stream) timed out after {}s", step, timeout_seconds)
        raise
