# tools/eval_retrieval.py
"""
[方案8] 离线检索评测脚本（调参 / GridSearch 辅助）。

用法：
    # 仅校验数据集格式（无需启动 Milvus / Mongo，可在任意环境自检）
    python tools/eval_retrieval.py --queries eval_queries.json --validate-only

    # 跑完整评测（需先启动 Milvus / MongoDB / 嵌入模型，并用项目 venv）
    python tools/eval_retrieval.py --queries eval_queries.json --top-k 5 --out eval_report.json

eval_queries.json 格式：
[
  {"query": "H3C ER2100 默认管理地址", "keywords": ["192.168.1.1", "admin"]},
  {"query": "万用表怎么测电压",        "keywords": ["表笔", "电压档"]}
]

每条 query 跑一遍完整的 KBQueryWorkflowV2（非流式），统计：
  - 命中率（hit_rate）：答案或重排文档中是否包含任一 keyword
  - Recall@K：重排结果前 K 条内是否命中 keyword
  - 平均耗时 / P95 耗时
结合 config/retrieval_config.py 的阈值（min_cosine_score / item_confirm_* /
search_*_weight / rrf_hyde_weight），在不同参数下多次运行本脚本即可对比
Recall@K 与耗时，定位最优调参组合。
"""
import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 注意：KBQueryWorkflowV2 / task_utils 依赖 torch、milvus 等重依赖，
# 仅在「完整评测」时才需要。这里故意不在模块顶层 import，
# 使 --validate-only 模式（仅校验数据集 + 打印配置）可在任意环境自检，
# 不依赖 embedding / Milvus 是否就绪。


def _active_tuning_config() -> dict:
    """尽力读取当前生效的检索调参配置，便于报告自说明；读取失败返回空字典。"""
    try:
        from config.retrieval_config import retrieval_config as rc
        return {
            "min_cosine_score": rc.min_cosine_score,
            "item_confirm_high": rc.item_confirm_high,
            "item_confirm_mid": rc.item_confirm_mid,
            "search_dense_weight": rc.search_dense_weight,
            "search_sparse_weight": rc.search_sparse_weight,
            "rrf_hyde_weight": rc.rrf_hyde_weight,
        }
    except Exception:
        return {}


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
    """懒加载检索链路（重依赖），仅在完整评测时调用。"""
    from processor.query_processor.main_graph_v2 import KBQueryWorkflowV2
    from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING
    update_task_status(session_id, TASK_STATUS_PROCESSING, False)
    workflow = KBQueryWorkflowV2()
    return workflow.run(_build_init_state(session_id, query), stream=False)


def _hit_in_topk(docs: list, keywords: list, k: int) -> bool:
    """前 min(K, len(docs)) 条文档中是否包含任一 keyword。"""
    top = docs[:k] if k and k > 0 else docs
    blob = " ".join((d.get("content") or "") for d in top)
    return any(kw.lower() in blob.lower() for kw in keywords)


def _validate(dataset_path: str) -> list:
    """校验数据集格式，返回用例列表；失败抛异常（沙箱可自检）。"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError("数据集应为非空 JSON 数组")
    for i, c in enumerate(cases, 1):
        if not isinstance(c, dict) or not c.get("query"):
            raise ValueError(f"第 {i} 条缺少 query 字段")
        if not isinstance(c.get("keywords", []), list):
            raise ValueError(f"第 {i} 条 keywords 应为数组")
    return cases


def main():
    ap = argparse.ArgumentParser(description="离线检索评测（Recall@K / 耗时 / 调参）")
    ap.add_argument("--queries", required=True, help="评测 query JSON 路径")
    ap.add_argument("--out", default="", help="评测报告输出 JSON 路径（可选）")
    ap.add_argument("--top-k", type=int, default=5, help="Recall@K 的 K 值（默认 5）")
    ap.add_argument("--validate-only", action="store_true",
                    help="仅校验数据集格式并打印配置，不运行检索链路（无需 Docker）")
    args = ap.parse_args()

    cases = _validate(args.queries)
    cfg = _active_tuning_config()

    print(f"数据集：{args.queries}  用例数：{len(cases)}")
    if cfg:
        print("当前调参配置：" + ", ".join(f"{k}={v}" for k, v in cfg.items()))

    if args.validate_only:
        print("校验通过（--validate-only 模式未运行检索链路）")
        return

    hits = 0
    hits_k = 0
    total = len(cases)
    costs = []
    results = []
    for i, case in enumerate(cases, 1):
        q = case.get("query", "")
        kws = case.get("keywords", [])
        sid = f"eval_{i}_{int(time.time() * 1000)}"
        t0 = time.time()
        try:
            state = _run_one(q, sid)
            err = ""
        except Exception as e:  # 单条失败不影响整体统计
            state = {"answer": "", "reranked_docs": []}
            err = repr(e)
        cost = time.time() - t0
        costs.append(cost)

        answer = state.get("answer") or ""
        docs = state.get("reranked_docs") or []
        blob = answer + " " + " ".join((d.get("content") or "") for d in docs)
        if kws:
            hit = any(kw.lower() in blob.lower() for kw in kws)
        else:
            hit = len(docs) > 0
        hit_k = _hit_in_topk(docs, kws, args.top_k) if kws else (len(docs) > 0)
        hits += 1 if hit else 0
        hits_k += 1 if hit_k else 0
        results.append({"query": q, "hit": hit, "hit@k": hit_k,
                        "cost_s": round(cost, 3), "docs": len(docs), "err": err})
        flag = "ERR" if err else f"hit={hit} hit@{args.top_k}={hit_k}"
        print(f"[{i}/{total}] {flag} cost={cost:.2f}s q={q[:30]}")

    avg = sum(costs) / total if total else 0.0
    costs_sorted = sorted(costs)
    p95_idx = min(total - 1, max(0, int(total * 0.95))) if costs_sorted else 0
    p95 = costs_sorted[p95_idx] if costs_sorted else 0.0
    print(f"\n命中率 hit_rate = {hits}/{total} = {hits / total:.2%}")
    print(f"Recall@{args.top_k} = {hits_k}/{total} = {hits_k / total:.2%}")
    print(f"平均耗时 = {avg:.2f}s   P95 耗时 = {p95:.2f}s")

    if args.out:
        report = {
            "dataset": args.queries,
            "total": total,
            "top_k": args.top_k,
            "hit_rate": hits / total,
            "recall_at_k": hits_k / total,
            "avg_cost_s": round(avg, 3),
            "p95_cost_s": round(p95, 3),
            "tuning_config": cfg,
            "details": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已写出：{args.out}")


if __name__ == "__main__":
    main()
