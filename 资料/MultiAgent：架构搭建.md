架构设计
1. Master Agent：是有状态的。和用户进行多轮对话，数据收集。把对应的数据传给其他agent进行一次调用即可
2. Worker Agent：都应该是无状态；
   核心思路：每个 Worker Agent 被包装成一个 @tool，Supervisor 通过调用这些工具来完成任务委派。Supervisor 负责意图识别和对话管理，Worker 只负责专项任务。
联动机制
   Supervisor 调用 Worker 的完整流程：

3. Worker Agent 实现：暂时模拟功能
   每个 Worker 用 create_agent 创建，当前阶段功能用模拟数据代替，重点验证联动流程。
1. 智慧问诊 Agent
# src/agents/workers/inquiry_agent.py

from langchain.agents import create_agent
from langchain_core.tools import tool
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

2. 报告解读 Agent
# src/agents/workers/report_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

REPORT_SYSTEM_PROMPT = """你是天宫医疗的报告解读助手。

你的职责：
1. 解读患者上传的检验报告、影像报告
2. 提取关键异常指标，用通俗语言解释含义
3. 标注需要重点关注的项目
4. 给出初步建议（是否需要复查、就诊等）

回复格式：
- 报告类型：xxx
- 关键发现：xxx
- 异常指标：xxx（正常范围：xxx，当前值：xxx）
- 建议：xxx

注意：解读结果仅供参考，最终诊断以医生为准。"""


def create_report_agent():
llm = get_llm(temperature=0.1)

    # 当前阶段：模拟解读
    # 未来可添加：Qwen-VL 视觉模型工具、历史报告对比工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=REPORT_SYSTEM_PROMPT,
        name="report_agent",
    )


_report_agent = None

def get_report_agent():
global _report_agent
if _report_agent is None:
_report_agent = create_report_agent()
return _report_agent

3. 药物 Agent
# src/agents/workers/drug_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

DRUG_SYSTEM_PROMPT = """你是天宫医疗的药物咨询助手。

你的职责：
1. 根据患者病情推荐合适的药物
2. 检测药物之间的相互作用（药物交互检测）
3. 审查处方，找出潜在风险（剂量、禁忌症、过敏史冲突）
4. 用通俗语言解释用药注意事项

回复格式：
- 推荐药物：xxx（用途：xxx，用法：xxx）
- 药物交互风险：xxx
- 注意事项：xxx
- 禁忌提示：xxx

注意：药物建议仅供参考，请遵医嘱用药。"""


def create_drug_agent():
llm = get_llm(temperature=0.1)

    # 当前阶段：模拟推荐
    # 未来可添加：Neo4j 知识图谱查询工具（NL2Cypher）、药物数据库检索工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=DRUG_SYSTEM_PROMPT,
        name="drug_agent",
    )


_drug_agent = None

def get_drug_agent():
global _drug_agent
if _drug_agent is None:
_drug_agent = create_drug_agent()
return _drug_agent
4. 知识问答 Agent
# src/agents/workers/knowledge_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

KNOWLEDGE_SYSTEM_PROMPT = """你是天宫医疗的医学知识助手。

你的职责：
1. 回答患者关于疾病、症状、治疗方案的咨询
2. 辅助医生进行医学文献检索
3. 提供循证医学参考（治疗指南、研究进展）
4. 解释医学术语，让患者易于理解

回复格式：
- 问题解答：xxx
- 相关知识：xxx
- 参考来源：xxx（如有）
- 延伸阅读：xxx（如有）

注意：知识问答不替代专业医疗建议，如有疑虑请咨询医生。"""


def create_knowledge_agent():
llm = get_llm(temperature=0.5)

    # 当前阶段：纯 LLM 知识回答
    # 未来可添加：RAG 检索工具（LlamaIndex）、HyDE 增强检索、GraphRAG 工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )


_knowledge_agent = None

def get_knowledge_agent():
global _knowledge_agent
if _knowledge_agent is None:
_knowledge_agent = create_knowledge_agent()
return _knowledge_agent

5. 运营数据 Agent
# src/agents/workers/operation_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

OPERATION_SYSTEM_PROMPT = """你是天宫医疗的运营数据助手。

你的职责：
1. 根据运营人员的自然语言问题，查询统计数据库中的运营数据
2. 生成数据报表和趋势分析
3. 辅助医院管理层做运营决策

重要安全规则：
- 只允许查询聚合统计数据，严禁返回患者个人信息
- 所有查询结果需脱敏处理
- 如果问题涉及患者隐私，拒绝回答并说明原因

回复格式：
- 数据摘要：xxx
- 关键指标：xxx
- 趋势分析：xxx
- 决策建议：xxx

注意：数据查询结果仅供内部运营参考。"""


def create_operation_agent():
llm = get_llm(temperature=0.2)

    # 当前阶段：模拟数据返回
    # 未来可添加：NL2SQL 工具（查询 PostgreSQL）、数据可视化工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=OPERATION_SYSTEM_PROMPT,
        name="operation_agent",
    )


_operation_agent = None

def get_operation_agent():
global _operation_agent
if _operation_agent is None:
_operation_agent = create_operation_agent()
return _operation_agent


4. 将 Worker 包装为工具
   每个 Worker Agent 被包装成 Supervisor 可以调用的工具。
# src/agents/worker_tools.py

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
5. Supervisor Agent 完整实现
# src/agents/supervisor_agent.py

import redis.asyncio as aioredis
from langgraph_checkpoint_redis import AsyncRedisSaver
from langchain.agents import create_agent
from src.core.config import get_llm

from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.agents.memory_tools import save_memory, search_memory
from src.agents.worker_tools import WORKER_TOOLS

settings = get_settings()

SUPERVISOR_SYSTEM_PROMPT = """你是天宫医疗的智能总助手（Supervisor）。

你的核心职责：
1. 与患者/医生/运营人员进行多轮对话
2. 准确识别用户意图，将任务分派给合适的专项助手
3. 整合专项助手的结果，给出清晰、友好的最终回复
4. 主动收集必要信息（如症状描述不清时追问）
5. 管理对话上下文，保持对话连贯性

可调用的专项助手：
- call_inquiry_agent：智慧问诊（症状分诊、挂号建议）
- call_report_agent：报告解读（检验单、影像报告）
- call_drug_agent：药物咨询（用药推荐、药物交互、处方审查）
- call_knowledge_agent：医学知识问答（疾病科普、治疗方案）
- call_operation_agent：运营数据查询（仅限内部运营人员）

记忆工具：
- save_memory：将重要信息（病史、过敏史、用药偏好等）保存到长期记忆
- search_memory：从长期记忆中检索用户历史信息

工作原则：
- 优先从长期记忆中检索用户历史信息，避免重复询问
- 遇到复杂问题可以串联多个专项助手（先问诊再查药）
- 始终以患者安全为第一优先级
- 对话语气温和、专业、易懂"""

# 剩余忽略：主要是提示词和工具
# ── 工具 = 记忆工具 + Worker 工具 ─────────────────────────────────
tools = [save_memory, search_memory] + WORKER_TOOLS

6. 验证测试
1. 单独测试 Worker Agent
# 在 Python 交互环境或测试脚本中运行
import asyncio
from src.agents.workers.inquiry_agent import get_inquiry_agent

async def test_inquiry():
agent = get_inquiry_agent()
result = await agent.ainvoke({
"messages": [{"role": "user", "content": "我头疼发烧两天了，体温38.5度"}]
})
print(result["messages"][-1].content)
2. 测试 Supervisor 调用 Worker
   import asyncio
   from src.agents.supervisor_agent import create_supervisor_agent


async def test_supervisor():
agent = await create_supervisor_agent()
config = {"configurable": {"thread_id": "test_001"}}

    # 第一轮：问诊场景 → 应触发 call_inquiry_agent
    r1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "我最近头疼发烧，不知道该挂什么科"}]},
        config=config,
    )
    print("=== 第一轮（问诊）===")
    print(r1["messages"][-1].content)

    # 第二轮：药物场景 → 应触发 call_drug_agent
    r2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "医生开了布洛芬和阿莫西林，可以一起吃吗？"}]},
        config=config,
    )
    print("=== 第二轮（药物）===")
    print(r2["messages"][-1].content)

    # 第三轮：知识问答 → 应触发 call_knowledge_agent
    r3 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "感冒一般多久能好？"}]},
        config=config,
    )
    print("=== 第三轮（知识）===")
    print(r3["messages"][-1].content)
nt 创建，当前阶段功能用模拟数据代替，重点验证联动流程。