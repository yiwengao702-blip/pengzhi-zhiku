#!/usr/bin/env bash
# tools/run_eval.sh —— 一键拉起中间件并跑离线评测（调参辅助）
#
# 前置：本机已安装 Docker（Desktop 或 Engine），且在项目根目录执行本脚本。
# 本脚本只负责：① docker compose 拉起 Milvus/MongoDB/MinIO  ② 等待全部 healthy
#           ③ 用项目 venv 运行 tools/eval_retrieval.py。
# 评测本身还需要：.env 里配置好 LLM API Key + BGE_M3_PATH 指向本地模型 + 中间件可达。
#
# 用法：
#   bash tools/run_eval.sh                 # 默认 top-k=5 单组评测，报告写 eval_report.json
#   TOP_K=10 bash tools/run_eval.sh        # 自定义 top-k
#   NO_UP=1 bash tools/run_eval.sh         # 中间件已在跑，跳过 docker compose up
#   GRID=1 bash tools/run_eval.sh          # 网格搜索模式，对比表写 grid_result.json(+.csv)
#   GRID=1 GRID_DENSE="1.0,0.6" GRID_HYDE="0.8,0.4" GRID_MIN_SCORE="0.75,0.6" \
#     GRID_OUT=grid_result.json bash tools/run_eval.sh   # 自定义网格候选轴

set -euo pipefail

# 切到仓库根目录（无论从哪里调用）
cd "$(dirname "$0")/.."

# ---------- 1) 拉起中间件 ----------
if [ "${NO_UP:-0}" != "1" ]; then
  echo "[run_eval] docker compose up -d ..."
  docker compose up -d
else
  echo "[run_eval] 跳过 docker compose（NO_UP=1）"
fi

# ---------- 2) 等待全部服务 healthy ----------
echo "[run_eval] 等待中间件就绪 ..."
TRIES=40
for i in $(seq 1 $TRIES); do
  # 取每个服务的健康状态（compose ps 模板字段 .Health）
  HEALTHS=$(docker compose ps --format '{{.Health}}' 2>/dev/null | grep -v '^$' || true)
  UNHEALTHY=$(echo "$HEALTHS" | grep -v -E '^healthy$' || true)
  if [ -z "$UNHEALTHY" ] && [ -n "$HEALTHS" ]; then
    echo "[run_eval] 全部服务 healthy ✓"
    break
  fi
  echo "  等待中 ($i/$TRIES) ..."
  sleep 5
done

# ---------- 3) 选定 venv 解释器 ----------
if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python
fi

TOP_K="${TOP_K:-5}"

if [ "${GRID:-0}" = "1" ]; then
  # 网格搜索模式：运行时遍历权重组合并生成对比表（JSON + CSV）
  GRID_DENSE="${GRID_DENSE:-1.0,0.8,0.6,0.4,0.2}"
  GRID_HYDE="${GRID_HYDE:-1.0,0.8,0.6,0.4,0.2,0.0}"
  GRID_MIN_SCORE="${GRID_MIN_SCORE:-0.75,0.70,0.65}"
  GRID_OUT="${GRID_OUT:-grid_result.json}"
  echo "[run_eval] 网格搜索模式（top-k=$TOP_K，候选轴 dense=[$GRID_DENSE] hyde=[$GRID_HYDE] min=[$GRID_MIN_SCORE]）..."
  "$PY" tools/eval_retrieval.py \
    --queries tools/eval_queries.json \
    --top-k "$TOP_K" \
    --grid-search \
    --grid-dense "$GRID_DENSE" \
    --grid-hyde "$GRID_HYDE" \
    --grid-min-score "$GRID_MIN_SCORE" \
    --grid-out "$GRID_OUT"
  echo "[run_eval] 完成，对比报告见 $GRID_OUT（同目录含 grid_result.csv）"
else
  # 单组评测模式
  echo "[run_eval] 运行评测（top-k=$TOP_K）..."
  "$PY" tools/eval_retrieval.py \
    --queries tools/eval_queries.json \
    --top-k "$TOP_K" \
    --out eval_report.json
  echo "[run_eval] 完成，报告见 eval_report.json"
fi
