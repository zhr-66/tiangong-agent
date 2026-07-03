"""测试路由和改写覆盖所有场景"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.embeddings import DashScopeEmbeddings
from src.core.config import get_settings, get_llm
from src.agents.knowledge.query_rewriter import rewrite_query
from src.agents.knowledge.prompts import ROUTE_PROMPT
from langchain_core.messages import SystemMessage
from src.agents.llm_utils import ainvoke_with_timeout
import json

settings = get_settings()

# 测试用例： (提问, 期望route, 期望rewrite包含关键词)
TEST_CASES = [
    # doc_rag 场景
    ("阿莫西林胶囊的适应症和用法用量是什么", "doc_rag", "阿莫西林"),
    ("高血压的疾病描述和病因是什么", "doc_rag", "高血压"),
    ("高血压诊疗指南推荐的一线用药", "doc_rag", "高血压"),

    # graph_rag 场景
    ("高血压的常用药有哪些", "graph_rag", "高血压"),
    ("糖尿病有哪些常见症状", "graph_rag", "糖尿病"),
    ("头痛可能是什么疾病需要做什么检查", "graph_rag", "头痛"),

    # multi 场景
    ("高血压的病因和常用药", "multi", "高血压"),
    ("糖尿病合并高血压的患者能用哪些感冒药", "multi", "糖尿病"),
    ("高血压的治疗方式和推荐药", "multi", "高血压"),

    # doc_rag 多维度（都是文档类信息）
    ("高血压的病因和预防措施", "doc_rag", "高血压"),

    # prescription 场景
    ("帮我审核这张处方：阿莫西林0.5g tid + 甲硝唑0.4g bid", "prescription", "阿莫西林"),
    ("维C银翘片和氨氯地平能一起吃吗", "prescription", "维C银翘片"),

    # nl2sql 场景
    ("上个月各科室的问诊量排名", "nl2sql", "科室"),
    ("库存不足100的OTC药品有哪些", "nl2sql", "库存"),
]


async def route_query(llm, question: str) -> str:
    prompt = ROUTE_PROMPT.format(question=question)
    try:
        response = await ainvoke_with_timeout(
            llm, [SystemMessage(content=prompt)], step="test.route",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        return result.get("route", "unknown")
    except Exception as e:
        return f"error: {e}"


async def test_one(llm, question, expected_route, expected_keyword):
    """测试单个用例"""
    # 1. 改写
    rewrite_result = await rewrite_query(question, llm)
    queries = rewrite_result.get("queries", [question])
    rewritten = queries[0] if queries else question

    # 2. 路由（用原始问题）
    actual_route = await route_query(llm, question)

    # 3. 检查
    route_ok = actual_route == expected_route
    rewrite_ok = expected_keyword.lower() in rewritten.lower()

    return {
        "question": question,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "rewritten": rewritten,
        "queries": queries,
        "route_ok": route_ok,
        "rewrite_ok": rewrite_ok,
        "keyword": expected_keyword,
    }


async def main():
    llm = get_llm(temperature=0.0)

    # 并发测试所有用例
    tasks = [
        test_one(llm, q, er, kw)
        for q, er, kw in TEST_CASES
    ]
    results = await asyncio.gather(*tasks)

    print("=" * 110)
    print(f"{'提问':<40} | {'期望':<15} | {'实际route':<15} | {'实际rewrite':<30} | 结果")
    print("=" * 110)

    pass_count = 0
    fail_count = 0
    for r in results:
        status = "[OK]" if r["route_ok"] and r["rewrite_ok"] else "[FAIL]"
        if r["route_ok"] and r["rewrite_ok"]:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{r['question']:<40} | {r['expected_route']:<15} | {r['actual_route']:<15} | {r['rewritten'][:28]:<30} | {status}")

    print("=" * 110)
    print(f"通过: {pass_count}/{len(TEST_CASES)}  失败: {fail_count}/{len(TEST_CASES)}")

    if fail_count > 0:
        print("\n" + "=" * 110)
        print("失败用例详细分析：")
        print("=" * 110)
        for r in results:
            if not (r["route_ok"] and r["rewrite_ok"]):
                print(f"\n提问: {r['question']}")
                print(f"  期望: route={r['expected_route']}, rewrite含'{r['keyword']}'")
                print(f"  实际: route={r['actual_route']}, rewrite='{r['rewritten']}'")
                print(f"  所有queries: {r['queries']}")
                if not r["route_ok"]:
                    print(f"  [FAIL] route error")
                if not r["rewrite_ok"]:
                    print(f"  [FAIL] rewrite lost keyword")


if __name__ == "__main__":
    asyncio.run(main())
