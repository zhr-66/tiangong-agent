"""
公共医学数据集初始化脚本。
从 HuggingFace 下载公共医学数据集，清洗后导入 Milvus 知识库。

用法：
    # 导入全部数据集
    python scripts/init_public_datasets.py

    # 只导入指定数据集
    python scripts/init_public_datasets.py --dataset cmirb
    python scripts/init_public_datasets.py --dataset dialogue
    python scripts/init_public_datasets.py --dataset medqa

依赖：
    pip install datasets
"""

import asyncio
import argparse
import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient
from loguru import logger

from src.core.config import get_settings
from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.doc_rag import COLLECTION_NAME

settings = get_settings()
BATCH_SIZE = 50


def _get_deps():
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    ensure_knowledge_collection(milvus_client)
    return milvus_client, embedding_model


async def _insert_texts(
    milvus_client: MilvusClient,
    embedding_model,
    texts: list[str],
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
) -> int:
    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )

    all_data = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = await embedding_model.aembed_documents(batch)
        for j, (text_content, emb) in enumerate(zip(batch, embeddings)):
            idx = i + j
            all_data.append({
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "category": category,
                "page_number": 0,
                "chunk_index": idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

    if all_data:
        for i in range(0, len(all_data), 1000):
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data[i:i + 1000],
            )
    return len(all_data)


async def download_cmirb(milvus_client, embedding_model):
    """
    下载 CMIRB/MedicalRetrieval 数据集。

    说明：Python 3.13 + 旧版 datasets 2.13.0 + 新版 pyarrow 19 不兼容
    （PyExtensionType 已移除），因此绕过 datasets 库，直接下载 corpus.jsonl
    并本地解析。

    数据来源：https://huggingface.co/datasets/CMIRB/MedicalRetrieval
    """
    import json
    from huggingface_hub import hf_hub_download

    print("[INFO] 下载 CMIRB/MedicalRetrieval 数据集（直接拉取 corpus.jsonl）...")
    corpus_path = hf_hub_download(
        repo_id="CMIRB/MedicalRetrieval",
        filename="corpus.jsonl",
        repo_type="dataset",
    )

    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "") or row.get("content", "")
            if text and len(text.strip()) > 20:
                texts.append(text.strip()[:2000])

    print(f"[INFO] CMIRB 共 {len(texts)} 条有效文本，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"cmirb_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"CMIRB医学检索语料_{i // chunk_size + 1}",
            doc_type="literature",
            category="医学文献",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] CMIRB 导入完成，共 {total} 条")
    return total


async def download_med_dialogue(milvus_client, embedding_model):
    """
    下载 Chinese-medical-dialogue-data 数据集。

    说明：Python 3.13 + 旧版 datasets 2.13.0 + 新版 pyarrow 19 不兼容
    （PyExtensionType 已移除），因此绕过 datasets 库，直接下载 外科.zip
    并本地解析 CSV 文件。

    数据来源：https://huggingface.co/datasets/BillGPT/Chinese-medical-dialogue-data
    """
    import csv
    import zipfile
    from huggingface_hub import hf_hub_download

    print("[INFO] 下载 Chinese-medical-dialogue-data 数据集（直接拉取 外科.zip）...")
    zip_path = hf_hub_download(
        repo_id="BillGPT/Chinese-medical-dialogue-data",
        filename="外科.zip",
        repo_type="dataset",
    )

    texts = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as f:
                # 尝试多种编码，中文 CSV 常见 GBK/UTF-8
                content = f.read()
                for enc in ("utf-8", "gbk", "gb18030"):
                    try:
                        text_content = content.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"  [WARN] 无法解码 {name}，跳过")
                    continue

                reader = csv.DictReader(text_content.splitlines())
                for row in reader:
                    q = (row.get("ask") or row.get("question") or "").strip()
                    a = (row.get("answer") or "").strip()
                    if q and a and len(q) > 5 and len(a) > 10:
                        text = f"问：{q}\n答：{a}"
                        texts.append(text[:2000])

    print(f"[INFO] 医患对话共 {len(texts)} 条有效记录，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"med_dialogue_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"中文医患对话_{i // chunk_size + 1}",
            doc_type="literature",
            category="医患对话",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] 医患对话导入完成，共 {total} 条")
    return total


async def download_medqa():
    """
    下载 MedQA 中文 4 选项数据集。

    说明：bigbio/med_qa 是脚本式数据集（med_qa.py），新版 datasets 库
    (>=2.14) 已不再支持执行 dataset scripts，且与新版 pyarrow (>=15) 存在
    PyExtensionType 兼容性问题。因此这里绕过 datasets 库，直接通过
    huggingface_hub 下载 data_clean.zip 并本地解析 JSONL 文件。

    数据来源：https://huggingface.co/datasets/bigbio/med_qa
    原始项目：https://github.com/jind11/MedQA
    """
    import json
    import zipfile
    import tempfile
    import requests

    print("[INFO] 下载 MedQA 数据集（直接拉取 data_clean.zip）...")

    # 直接通过镜像 URL 下载，绕过 huggingface_hub 元数据校验
    # 优先 hf-mirror.com（国内加速），失败则回退官方源
    mirror_urls = [
        "https://hf-mirror.com/datasets/bigbio/med_qa/resolve/main/data_clean.zip",
        "https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data_clean.zip",
    ]

    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "data_clean.zip")

    downloaded = False
    for url in mirror_urls:
        try:
            print(f"  [尝试] {url}")
            with requests.get(url, stream=True, timeout=30, verify=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                print(f"  [连接成功] 文件大小：{total / 1024 / 1024:.1f} MB")
                with open(zip_path, "wb") as f:
                    downloaded_bytes = 0
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        pct = (downloaded_bytes / total * 100) if total else 0
                        print(f"\r  [下载中] {pct:5.1f}% ({downloaded_bytes // 1024}KB)", end="", flush=True)
            print(f"\n  [OK] 下载完成")
            downloaded = True
            break
        except Exception as e:
            print(f"  [失败] {e}")
            continue

    if not downloaded:
        raise RuntimeError("所有镜像源下载失败，请检查网络连接或配置代理。")

    # 中文 4 选项子集在 zip 内的相对路径
    split_files = {
        "train": "data_clean/questions/Mainland/4_options/train.jsonl",
        "test": "data_clean/questions/Mainland/4_options/test.jsonl",
        "validation": "data_clean/questions/Mainland/4_options/dev.jsonl",
    }

    eval_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        # 校验文件存在
        names = set(zf.namelist())
        for split, rel in split_files.items():
            # zip 内路径分隔符可能为反斜杠，统一兼容
            rel_norm = rel.replace("/", os.sep)
            matched = rel if rel in names else (rel_norm if rel_norm in names else None)
            if matched is None:
                print(f"[WARN] zip 内未找到 {rel}，跳过 {split}")
                continue

            with zf.open(matched) as f:
                lines = f.read().decode("utf-8").splitlines()

            count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question = row.get("question", "")
                options = row.get("options", {})
                answer = row.get("answer", "")
                # 转为 bigbio_qa schema：choices 为选项值列表，answer 为列表
                choices = list(options.values()) if isinstance(options, dict) else list(options)
                if question:
                    records.append({
                        "question": question,
                        "choices": choices,
                        "answer": [answer] if answer else [],
                        "split": split,
                    })
                    count += 1
            print(f"  [进度] {split}: {count} 条")

    output_path = os.path.join(eval_dir, "medqa_zh.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] MedQA 保存完成：{output_path}，共 {len(records)} 道题")
    return len(records)


async def main():
    parser = argparse.ArgumentParser(description="公共医学数据集初始化")
    parser.add_argument(
        "--dataset",
        choices=["cmirb", "dialogue", "medqa", "all"],
        default="all",
        help="要导入的数据集（默认 all）",
    )
    args = parser.parse_args()

    milvus_client, embedding_model = _get_deps()

    if args.dataset in ("cmirb", "all"):
        await download_cmirb(milvus_client, embedding_model)

    if args.dataset in ("dialogue", "all"):
        await download_med_dialogue(milvus_client, embedding_model)

    if args.dataset in ("medqa", "all"):
        await download_medqa()

    print("\n[DONE] 数据集初始化完成！")


if __name__ == "__main__":
    asyncio.run(main())