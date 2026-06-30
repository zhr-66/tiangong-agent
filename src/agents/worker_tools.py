from langchain_core.tools import tool
from src.agents.workers.inquiry_agent import get_inquiry_agent
from src.agents.workers.report_agent import get_report_agent
from src.agents.workers.drug_agent import get_drug_agent
from src.agents.workers.knowledge_agent import get_knowledge_agent
from src.agents.workers.operation_agent import get_operation_agent


@tool
async def call_inquiry_agent(message: str) -> str:
    """
    调用智慧问诊Agent，对患者进行智能分诊。
    适用场景：患者描述症状、询问挂哪个科室、需要预约挂号时。
    message: 患者描述的症状或问诊需求。
    """
    agent = get_inquiry_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_report_agent(message: str) -> str:
    """
    调用报告解读Agent，解读患者的检验报告或影像报告。
    适用场景：患者上传了报告需要解读、询问报告中某项指标含义时。
    message: 报告内容描述或患者的具体问题。
    """
    agent = get_report_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_drug_agent(message: str) -> str:
    """
    调用药物Agent，进行药物推荐、药物交互检测或处方审查。
    适用场景：询问用什么药、多种药物能否同服、处方是否安全时。
    message: 患者的用药问题或处方信息。
    """
    agent = get_drug_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_knowledge_agent(message: str) -> str:
    """
    调用知识问答Agent，回答医学知识类问题。
    适用场景：询问疾病知识、治疗方案、医学术语解释、文献检索时。
    message: 患者或医生的医学知识问题。
    """
    agent = get_knowledge_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_operation_agent(message: str) -> str:
    """
    调用运营数据Agent，查询医院运营统计数据。
    适用场景：运营人员询问就诊量、收入、科室排名等运营数据时。
    message: 运营人员的数据查询需求（自然语言）。
    """
    agent = get_operation_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


# 所有 Worker 工具列表，供 Supervisor 使用
WORKER_TOOLS = [
    call_inquiry_agent,
    call_report_agent,
    call_drug_agent,
    call_knowledge_agent,
    call_operation_agent,
]
