#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
RUNNER="${ROOT}/c2kv_eval/portable/ops/run_tau2_p6k_8baselines_4cards.sh"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/crossdataset_racer_gpu123_$(date +%Y%m%d_%H%M%S)}"
TAU2_ROOT="${TAU2_ROOT:-/home/zhuyuhan/project/benchmarks/tau2-bench}"
METHODS="c2kv,c2kv_racer,streamingllm_r25,streamingllm_r25_racer,h2o_r25,h2o_r25_racer,snapkv_r25,snapkv_r25_racer,pyramidkv_r25,pyramidkv_r25_racer"
mkdir -p "${OUT_ROOT}/logs"
PIDS=()
cleanup() {
  set +e
  for p in "${PIDS[@]:-}"; do kill -TERM "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]:-}"; do wait "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM
echo "GPU1=tau2 airline50; GPU2=BFCL multi_turn_miss_param; GPU3=BFCL multi_turn_long_context"
env BENCHMARK=tau2 DEVICES=1 PORTS=35810 PROXY_PORTS=36810 RUN_METHODS="${METHODS}" MAX_TASKS=50 TAU2_TASK_SET=airline TAU2_ROOT="${TAU2_ROOT}" RUN_ROOT="${OUT_ROOT}/tau2" bash "${RUNNER}" >"${OUT_ROOT}/logs/tau2.log" 2>&1 & PIDS+=("$!")
env BENCHMARK=bfcl DEVICES=2 PORTS=35820 PROXY_PORTS=36820 RUN_METHODS="${METHODS}" MAX_TASKS=200 BFCL_CATEGORY=multi_turn_miss_param RUN_ROOT="${OUT_ROOT}/bfcl_miss_param" bash "${RUNNER}" >"${OUT_ROOT}/logs/bfcl_miss_param.log" 2>&1 & PIDS+=("$!")
env BENCHMARK=bfcl DEVICES=3 PORTS=35830 PROXY_PORTS=36830 RUN_METHODS="${METHODS}" MAX_TASKS=200 BFCL_CATEGORY=multi_turn_long_context RUN_ROOT="${OUT_ROOT}/bfcl_long_context" bash "${RUNNER}" >"${OUT_ROOT}/logs/bfcl_long_context.log" 2>&1 & PIDS+=("$!")
status=0
for p in "${PIDS[@]}"; do wait "$p" || status=1; done
(( status == 0 )) || { echo "one or more datasets failed; see ${OUT_ROOT}/logs" >&2; exit 1; }
python - "${OUT_ROOT}" <<'PYEND'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for dataset_dir in ("tau2","bfcl_miss_param","bfcl_long_context"):
 for p in sorted((root/dataset_dir/"compare").glob("*/*/summary_*.json")):
  x=json.loads(p.read_text())
  rows.append({"dataset":dataset_dir,"arm":x.get("arm"),"official_score":x.get("official_score"),"official_correct_count":x.get("official_correct_count"),"official_total_count":x.get("official_total_count"),"summary_path":str(p)})
out=root/"cross_dataset_racer_summary.csv"
with out.open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["dataset","arm"]); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
PYEND
echo "completed; results: ${OUT_ROOT}"
