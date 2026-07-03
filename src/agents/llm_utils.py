from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


DEFAULT_LLM_TIMEOUT_SECONDS = 45


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
