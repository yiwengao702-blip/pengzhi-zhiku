# tools/eval_retrieval.py
"""
[P1/P2-方案8] 离线检索评测脚本（GridSearch 辅助）。

用法：
    python tools/eval_retrieval.py --queries eval_queries.json

eval_queries.json 格式：
[
  {"query": "H3C ER2100 默认管理地址", "keywords": ["192.168.1.1", "admin"]},
  {"query": "万用表怎么测电压",        "keywords": ["表笔", "电压档"]}
]

脚本对每条 query 跑一遍完整的 KBQueryWorkflowV2（非流式），统计：
  - 命中率（hit_rate）：答案或重排文档中是否包含任一 keyword
  - 平均耗时
便于在不同 ranker_weights / 阈值（见 .env + config/retrieval_config.py）下对比 Recall@K，指导调参。
注意：需先启动 Milvus / MongoDB / 嵌入模型，并用项目 venv 运行。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from processor.query_processor.main_graph_v2 import KBQueryWorkflowV2
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result


def _build_init_state(session_id: str, query: str) -> dict:
    return {
        "session_id": session_id,
        "message_id": "",
        "original_query": query,
        "embedding_chunks": [], "hyde_embedding_chunks": [], "hyde_doc": "",
        "web_search_docs": [], "rrf_chunks": [], "reranked_docs": [],
        "query_dense": None, "query_sparse": None, "hyde_dense": None, "hyde_sparse": None,
        "prompt": "", "answer": "", "item_names": [], "rewritten_query": "",
        "history": [], "is_stream": False,
    }


def _run_one(query: str, session_id: str) -> dict:
    workflow = KBQueryWorkflowV2()
    return workflow.run(_build_init_state(session_id, query), stream=False)


def main():
    ap = argparse.ArgumentParser(description="离线检索评测（Recall@K / 耗时）")
    ap.add_argument("--queries", required=True, help="评测 query JSON 路径")
    args = ap.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        cases = json.load(f)

    hits = 0
    total = len(cases)
    cost_sum = 0.0
    for i, case in enumerate(cases, 1):
        q = case.get("query", "")
        kws = case.get("keywords", [])
        sid = f"eval_{i}_{int(time.time() * 1000)}"
        update_task_status(sid, TASK_STATUS_PROCESSING, False)
        t0 = time.time()
        state = _run_one(q, sid)
        cost = time.time() - t0
        cost_sum += cost

        answer = state.get("answer") or ""
        blob = answer + " " + " ".join(
            (d.get("content") or "") for d in (state.get("reranked_docs") or [])
        )
        if kws:
            hit = any(kw.lower() in blob.lower() for kw in kws)
        else:
            # 无 keyword 时，以是否召回到文档作为命中
            hit = len(state.get("reranked_docs") or []) > 0
        hits += 1 if hit else 0
        print(f"[{i}/{total}] hit={hit} cost={cost:.2f}s q={q[:30]}")

    print(f"\n命中率 hit_rate = {hits}/{total} = {hits / total:.2%}   平均耗时 = {cost_sum / total:.2f}s")


if __name__ == "__main__":
    main()
