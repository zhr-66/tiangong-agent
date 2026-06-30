from langchain.agents import create_agent
from src.core.config import get_llm

INQUIRY_SYSTEM_PROMPT = """你是天宫医疗的智慧问诊助手。

你的职责：
1. 根据患者描述的症状，进行智能分诊
2. 判断患者应该挂哪个科室
3. 如有必要，主动追问关键症状信息
4. 评估病情紧急程度（紧急/普通/可预约）

回复格式：
- 分诊科室：xxx科
- 紧急程度：xxx
- 建议：xxx

注意：你只负责分诊建议，不做最终诊断。"""


def create_inquiry_agent():
    llm = get_llm(temperature=0.3)

    # 当前阶段：无工具，纯 LLM 推理
    # 未来可添加：患者历史记录查询工具、HyDE 检索工具、预约挂号工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=INQUIRY_SYSTEM_PROMPT,
        name="inquiry_agent",
    )


# 模块级单例
_inquiry_agent = None

def get_inquiry_agent():
    global _inquiry_agent
    if _inquiry_agent is None:
        _inquiry_agent = create_inquiry_agent()
    return _inquiry_agent
