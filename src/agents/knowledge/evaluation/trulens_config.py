"""
TruLens 会话配置 + LLM Provider + Dashboard 启动
"""
import hashlib
from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings

settings = get_settings()


def get_trulens_session() -> TruSession:
    """评估结果存入独立的 PostgreSQL 数据库"""
    return TruSession(
        database_url=f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
    )


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider（LLM-as-Judge）。
    通过 LiteLLM 对接 DashScope OpenAI 兼容接口（qwen-plus）。

    说明：TruLens feedback 函数默认使用 response_format 参数要求结构化输出，
    DeepSeek API 不支持该参数类型（返回 "This response_format type is unavailable now"），
    而 DashScope 的 OpenAI 兼容接口支持 response_format，因此评估用 LLM 改用 qwen-plus。
    """
    return LiteLLM(
        model_engine="openai/qwen-plus",
        completion_kwargs={
            "api_key": settings.DASHSCOPE_API_KEY,
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    )


def launch_dashboard(port: int = 8501):
    """启动 TruLens Streamlit Dashboard"""
    from trulens.dashboard import run_dashboard
    session = get_trulens_session()
    run_dashboard(session=session, port=port)



def should_trace(user_id: str, sample_rate: float = 0.01) -> bool:
    """按用户 ID 哈希采样"""
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return (h % 10000) < (sample_rate * 10000)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    launch_dashboard(port)