from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings

settings = get_settings()


def get_trulens_session() -> TruSession:
    """
    初始化 TruLens 会话。
    评估结果存入 PostgreSQL，与业务数据隔离。
    """
    session = TruSession(
        database_url=f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
    )
    return session


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider。
    通过 LiteLLM 对接 DashScope OpenAI 兼容接口（qwen-plus）。

    说明：TruLens 的 feedback 函数默认使用 response_format 参数要求结构化输出，
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


def launch_dashboard(session: TruSession | None = None, port: int = 8501):
    """启动 TruLens Dashboard（Streamlit 界面）。"""
    from trulens.dashboard import run_dashboard
    s = session or get_trulens_session()
    run_dashboard(session=s, port=port)