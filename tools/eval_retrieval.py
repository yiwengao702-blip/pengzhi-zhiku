# tools/eval_retrieval.py
"""
[方案8] 离线检索评测脚本（调参 / GridSearch 辅助）。

用法：
    # 仅校验数据集格式（无需启动 Milvus / Mongo，可在任意环境自检）
    python tools/eval_retrieval.py --queries eval_queries.json --validate-only

    # 跑完整评测（需先启动 Milvus / MongoDB / 嵌入模型，并用项目 venv）
    python tools/eval_retrieval.py --queries eval_queries.json --top-k 5 --out eval_report.json

    # 自动网格搜索权重组合并生成对比表（需 Docker + venv）
    python tools/eval_retrieval.py --queries eval_queries.json --grid-search \
        --grid-dense 1.0,0.8,0.6,0.4,0.2 --grid-hyde 1.0,0.8,0.6,0.4,0.2,0.0 \
        --grid-min-score 0.75,0.70,0.65 --grid-out grid_result.json

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

--grid-search 模式会在运行时改写 retrieval_config 的检索权重并立即重跑整组用例，
自动遍历 search_dense_weight / rrf_hyde_weight / min_cosine_score 的组合，
输出按 Recall@K（降序）/ 平均耗时（升序）排序的对比表（JSON + CSV）。
注意：search_sparse_weight 由 1 - search_dense_weight 推导，二者作为一个轴联动。
"""
import argparse
import csv
import itertools
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


def _parse_float_list(raw: str, default):
    """把逗号分隔的字符串解析为 float 列表；空串或解析失败回落到 default。"""
    if not raw:
        return list(default)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            print(f"  警告：忽略无法解析为数值的网格参数片段：{part!r}")
    return out if out else list(default)


def _patch_config(dense: float, hyde: float, min_score: float) -> None:
    """
    运行时改写检索权重配置，使 GridSearch 立即生效。

    关键前提：各检索节点在 process() 调用时实时读取 retrieval_config
    （search_*_weight、min_cosine_score、rrf_hyde_weight 均为 call-time 读取），
    因此这里直接改单例字段即可影响后续所有查询，无需重启进程。
    """
    from config.retrieval_config import retrieval_config as rc
    rc.search_dense_weight = float(dense)
    rc.search_sparse_weight = float(round(1.0 - dense, 6))  # 稀疏权重由稠密推导，二者联动
    rc.rrf_hyde_weight = float(hyde)
    rc.min_cosine_score = float(min_score)
    # 防御性同步：若 node_rrf 仍持有模块级常量（理论上已改为 call-time 读取），一并更新
    try:
        import processor.query_processor.nodes.node_rrf as nr
        if hasattr(nr, "RRF_HYDE_WEIGHT"):
            nr.RRF_HYDE_WEIGHT = float(hyde)
    except Exception:
        pass


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


def _run_one(query: str, session_id: str, workflow=None) -> dict:
    """懒加载检索链路（重依赖），仅在完整评测时调用。workflow 可复用以省去重复编译。"""
    from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING
    update_task_status(session_id, TASK_STATUS_PROCESSING, False)
    if workflow is None:
        # 仅在确实没有可复用实例时才加载重依赖（torch/milvus/dotenv 等），
        # 注入 workflow 时跳过，便于评测复用与单元测试桩。
        from processor.query_processor.main_graph_v2 import KBQueryWorkflowV2
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


def _evaluate_cases(cases, top_k, workflow=None):
    """
    对整组用例跑一遍检索链路，返回 (hits, hits_k, costs, total, per_case)。
    复用同一 workflow 实例以省去重复编译与模型加载。
    """
    hits = 0
    hits_k = 0
    total = len(cases)
    costs = []
    per_case = []
    for i, case in enumerate(cases, 1):
        q = case.get("query", "")
        kws = case.get("keywords", [])
        sid = f"eval_{i}_{int(time.time() * 1000)}"
        t0 = time.time()
        try:
            state = _run_one(q, sid, workflow=workflow)
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
        hit_k = _hit_in_topk(docs, kws, top_k) if kws else (len(docs) > 0)
        hits += 1 if hit else 0
        hits_k += 1 if hit_k else 0
        per_case.append({"query": q, "hit": hit, "hit@k": hit_k,
                         "cost_s": round(cost, 3), "docs": len(docs), "err": err})
    return hits, hits_k, costs, total, per_case


def _summarize(costs, total):
    """计算平均耗时 / P95 耗时。"""
    avg = sum(costs) / total if total else 0.0
    costs_sorted = sorted(costs)
    p95_idx = min(total - 1, max(0, int(total * 0.95))) if costs_sorted else 0
    p95 = costs_sorted[p95_idx] if costs_sorted else 0.0
    return avg, p95


def _run_grid(axes: dict, cases, top_k, out_path: str, workflow=None):
    """
    网格搜索：遍历 (search_dense_weight, rrf_hyde_weight, min_cosine_score) 组合，
    每组跑完所有用例后记录一行指标，最终按 Recall@K 降序、平均耗时升序排序输出对比表。
    workflow 可外部注入复用；缺省时首次调用会懒加载并构建 KBQueryWorkflowV2 实例。
    """
    dense_list = axes["dense"]
    hyde_list = axes["hyde"]
    minscore_list = axes["min_score"]
    combos = list(itertools.product(dense_list, hyde_list, minscore_list))
    total_combos = len(combos)
    print(f"\n=== GridSearch 启动：共 {total_combos} 组权重组合 × {len(cases)} 条用例 ===")

    # 复用同一 workflow 实例（仅构建 StateGraph，不加载模型），避免每组重编译/重加载模型。
    # 模型在各检索节点内部懒加载并缓存，首次查询后常驻，后续组合直接复用，显著提速。
    if workflow is None:
        from processor.query_processor.main_graph_v2 import KBQueryWorkflowV2
        workflow = KBQueryWorkflowV2()
    rows = []
    for ci, (d, h, m) in enumerate(combos, 1):
        _patch_config(d, h, m)
        hits, hits_k, costs, total, _per = _evaluate_cases(cases, top_k, workflow=workflow)
        avg, p95 = _summarize(costs, total)
        recall = hits_k / total if total else 0.0
        hit_rate = hits / total if total else 0.0
        row = {
            "combo_id": ci,
            "search_dense_weight": round(d, 4),
            "search_sparse_weight": round(1.0 - d, 4),
            "rrf_hyde_weight": round(h, 4),
            "min_cosine_score": round(m, 4),
            "hit_rate": round(hit_rate, 4),
            "recall_at_k": round(recall, 4),
            "avg_cost_s": round(avg, 3),
            "p95_cost_s": round(p95, 3),
        }
        rows.append(row)
        print(f"[{ci}/{total_combos}] dense={d} hyde={h} min={m} "
              f"=> Recall@{top_k}={recall:.2%} hit={hit_rate:.2%} avg={avg:.2f}s")

    # 排序：Recall@K 降序优先，其次平均耗时升序
    rows.sort(key=lambda r: (-r["recall_at_k"], r["avg_cost_s"]))
    best = rows[0] if rows else {}
    print(f"\n=== GridSearch 完成：最优组合按 Recall@{top_k} 排序 ===")
    print(f"  最佳：dense={best.get('search_dense_weight')} hyde={best.get('rrf_hyde_weight')} "
          f"min={best.get('min_cosine_score')} => Recall@{top_k}={best.get('recall_at_k'):.2%} "
          f"avg={best.get('avg_cost_s')}s")

    _emit_grid_report(rows, axes, best, cases, top_k, out_path)
    return rows


def _emit_grid_report(rows, axes, best, cases, top_k, out_path: str):
    """把对比表写成 JSON + CSV（CSV 与 JSON 同目录、同名不同扩展名）。"""
    report = {
        "mode": "grid-search",
        "total_cases": len(cases),
        "top_k": top_k,
        "axes": {
            "search_dense_weight": axes["dense"],
            "search_sparse_weight": [round(1.0 - d, 4) for d in axes["dense"]],
            "rrf_hyde_weight": axes["hyde"],
            "min_cosine_score": axes["min_score"],
        },
        "best": best,
        "grid": rows,
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"网格对比报告(JSON)已写出：{out_path}")
        # 附带一份 CSV，便于在表格软件里直接看
        csv_path = Path(out_path).with_suffix(".csv")
        fields = ["combo_id", "search_dense_weight", "search_sparse_weight",
                  "rrf_hyde_weight", "min_cosine_score", "hit_rate",
                  "recall_at_k", "avg_cost_s", "p95_cost_s"]
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"网格对比报告(CSV)已写出：{csv_path}")
        except Exception as e:
            print(f"  CSV 写出跳过（非致命）：{e}")
    else:
        # 无输出路径时仅打印表格
        print("\ncombo | dense | sparse | hyde | min_score | hit_rate | Recall@K | avg_s | p95_s")
        for r in rows:
            print(f"{r['combo_id']:>5} | {r['search_dense_weight']:.2f} | "
                  f"{r['search_sparse_weight']:.2f} | {r['rrf_hyde_weight']:.2f} | "
                  f"{r['min_cosine_score']:.2f} | {r['hit_rate']:.2%} | "
                  f"{r['recall_at_k']:.2%} | {r['avg_cost_s']:.2f} | {r['p95_cost_s']:.2f}")


def main():
    ap = argparse.ArgumentParser(description="离线检索评测（Recall@K / 耗时 / 调参 / GridSearch）")
    ap.add_argument("--queries", required=True, help="评测 query JSON 路径")
    ap.add_argument("--out", default="", help="评测报告输出 JSON 路径（可选，单组评测用）")
    ap.add_argument("--top-k", type=int, default=5, help="Recall@K 的 K 值（默认 5）")
    ap.add_argument("--validate-only", action="store_true",
                    help="仅校验数据集格式并打印配置，不运行检索链路（无需 Docker）")
    # ===== GridSearch 相关选项 =====
    ap.add_argument("--grid-search", action="store_true",
                    help="自动遍历权重组合并生成对比表（需 Docker + 项目 venv）")
    ap.add_argument("--grid-dense", default="1.0,0.8,0.6,0.4,0.2",
                    help="search_dense_weight 候选（逗号分隔），sparse 由 1-该值推导")
    ap.add_argument("--grid-hyde", default="1.0,0.8,0.6,0.4,0.2,0.0",
                    help="rrf_hyde_weight 候选（逗号分隔）")
    ap.add_argument("--grid-min-score", default="0.75,0.70,0.65",
                    help="min_cosine_score 候选（逗号分隔）")
    ap.add_argument("--grid-out", default="",
                    help="网格对比报告输出 JSON 路径（默认复用 --out）")
    args = ap.parse_args()

    if args.validate_only and args.grid_search:
        ap.error("--validate-only 与 --grid-search 互斥")

    cases = _validate(args.queries)
    cfg = _active_tuning_config()

    print(f"数据集：{args.queries}  用例数：{len(cases)}")
    if cfg:
        print("当前调参配置：" + ", ".join(f"{k}={v}" for k, v in cfg.items()))

    if args.validate_only:
        print("校验通过（--validate-only 模式未运行检索链路）")
        return

    if args.grid_search:
        axes = {
            "dense": _parse_float_list(args.grid_dense, [1.0, 0.8, 0.6, 0.4, 0.2]),
            "hyde": _parse_float_list(args.grid_hyde, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
            "min_score": _parse_float_list(args.grid_min_score, [0.75, 0.70, 0.65]),
        }
        grid_out = args.grid_out or args.out
        _run_grid(axes, cases, args.top_k, grid_out)
        return

    # ===== 单组评测（默认行为） =====
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
