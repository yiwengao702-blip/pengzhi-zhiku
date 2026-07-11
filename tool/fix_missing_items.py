"""
临时修复脚本：手动补全 kb_item_names 中缺失的商品记录
原因：导入流程在 node_document_split / node_item_name_recognition 阶段异常中断，
     导致大部分已上传文档的商品名未入库
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# 确保项目根路径在 sys.path 中，方便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import MilvusClient
from utils.embedding_utils import generate_embeddings

# ─── 1. 待补全的文档列表（去重后的 file_title → item_name） ───
# item_name 从文件名解析而来，去除空格/换行等格式字符（与 node_item_name_recognition 清洗逻辑一致）
MISSING_ITEMS = {
    # 有 chunks.json 但没进库的
    "迅饶网关与小米产品通讯配置说明": "迅饶网关与小米产品通讯配置说明",
    # 有 .md 但没 chunks.json 的（导入流程中断在切分之前）
    "H3C NER214W路由器 用户手册-6W101-整本手册": "H3CNER214W路由器",
    "H3C MSR-XS系列路由器 用户手册-R0821-6W100-整本手册": "H3CMSR-XS系列路由器",
    "HUAWEI MateBook B5-330 用户指南-(WRTDZ-WFH9&WFE9,Windows11_01,zh-cn)": "HUAWEIMateBookB5-330",
    "hak180使用说明书": "BrotherHAK180烫金机",
    "hak180产品安全手册": "BrotherHAK180烫金机",
    "Aolynk CC系列室内型Cable网络集中器 用户手册-6W202-整本手册": "AolynkCC系列Cable网络集中器",
    "H3C NR-1200W企业级无线宽带路由器 用户手册-5W100-整本手册": "H3CNR-1200W企业级无线宽带路由器",
    "联想至像大象系列打印机用户手册V1.02-20250217": "联想至像大象系列打印机",
    "H3C MER系列路由器 用户手册-R0821-6W105-整本手册": "H3CMER系列路由器",
    "联想海豚用户手册": "联想海豚",
    # 备注：LangGraph.md / LangGraph_new.md 是技术文档，非产品，跳过
}


def cleanse_item_name(name: str) -> str:
    """
    清洗 item_name：去除所有空白字符（与 node_item_name_recognition._step_3_call_llm 逻辑一致）
    确保存入 Milvus 后与查询阶段 LLM 提取出的格式完全一致
    """
    return name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")


def main():
    milvus_url = os.getenv("MILVUS_URL")
    collection_name = os.getenv("ITEM_NAME_COLLECTION", "kb_item_names")
    print(f"[INFO] 连接 Milvus: {milvus_url}")
    print(f"[INFO] 目标集合: {collection_name}")

    client = MilvusClient(milvus_url)

    # ─── 2. 查询已存在的 file_title，去重 ───
    try:
        existing = client.query(
            collection_name=collection_name,
            output_fields=["file_title"],
            limit=500
        )
        existing_titles = {r["file_title"] for r in existing}
        print(f"[INFO] 已有 {len(existing_titles)} 个已入库商品: {existing_titles}")
    except Exception:
        # 如果集合未加载或查询失败，尝试先加载
        print("[WARN] 直接查询失败，尝试加载集合...")
        client.load_collection(collection_name)
        existing = client.query(
            collection_name=collection_name,
            output_fields=["file_title"],
            limit=500
        )
        existing_titles = {r["file_title"] for r in existing}
        print(f"[INFO] 已有 {len(existing_titles)} 个已入库商品")

    # ─── 3. 过滤出真正需要插入的 ───
    to_insert = []
    for file_title, raw_name in MISSING_ITEMS.items():
        if file_title in existing_titles:
            print(f"[SKIP] 已存在: {file_title}")
            continue
        to_insert.append((file_title, cleanse_item_name(raw_name)))

    if not to_insert:
        print("[DONE] 没有需要补全的商品记录！")
        return

    print(f"\n[INFO] 待插入 {len(to_insert)} 条记录:")
    for ft, name in to_insert:
        print(f"  - {ft} → {name}")

    # ─── 4. 批量生成向量 ───
    print("\n[INFO] 开始生成 BGE-M3 向量...")
    item_names_for_embedding = [name for _, name in to_insert]
    embeddings = generate_embeddings(item_names_for_embedding)
    print(f"[INFO] 向量生成完成: dense={len(embeddings['dense'])} 条, sparse={len(embeddings['sparse'])} 条")

    # ─── 5. 批量插入 Milvus ───
    print("\n[INFO] 开始插入 Milvus...")
    success_count = 0
    for i, (file_title, item_name) in enumerate(to_insert):
        try:
            # 先删除同 file_title 的旧记录（幂等，与 node_item_name_recognition._step_6 一致）
            client.delete(
                collection_name=collection_name,
                filter=f"file_title=='{file_title}'"
            )
            # 插入新记录
            client.insert(
                collection_name=collection_name,
                data=[{
                    "file_title": file_title,
                    "item_name": item_name,
                    "dense_vector": embeddings["dense"][i],
                    "sparse_vector": embeddings["sparse"][i],
                }]
            )
            print(f"  [OK] {file_title} → {item_name}")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {file_title}: {e}")

    print(f"\n[DONE] 成功插入 {success_count}/{len(to_insert)} 条商品记录")


if __name__ == "__main__":
    main()
