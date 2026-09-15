#!/usr/bin/env bash
# User-operated only: this script DOES start a private NPU server when invoked.
set -Eeuo pipefail
ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
EXAMPLES="${EXAMPLES:-2}"
case "$EXAMPLES" in 2|52|200) ;; *) echo 'EXAMPLES must be 2, 52 or 200' >&2; exit 2;; esac
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/runs/bfcl_persistent_history_${EXAMPLES}_card6_$(date +%Y%m%d_%H%M%S)}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
task_ids_dir="$(mktemp -d /tmp/bfcl-history-ids.XXXXXX)"
trap 'rm -f "$task_ids_dir/ids.txt"; rmdir "$task_ids_dir"' EXIT
IDS_PATH=__NONE__
if [[ "$EXAMPLES" != 200 ]]; then
  FEATURES_CSV="${FEATURES_CSV:-/home/zhuyuhan/zh_restore/runs/detector_v2_regen/detector_cv/feature_benchmark/detector_features.csv}"
  "$BFCL_PYTHON" - "$FEATURES_CSV" "$task_ids_dir/ids.txt" <<'PY'
import csv, sys
from pathlib import Path
with open(sys.argv[1]) as f:
    ids = sorted({r['id'] for r in csv.DictReader(f)})
if len(ids) != 52 or any(not s.startswith('multi_turn_base_') for s in ids):
    raise RuntimeError('Expected the frozen stable52 feature source; refuse a different case set')
Path(sys.argv[2]).write_text('\n'.join(ids)+'\n')
PY
  IDS_PATH="$task_ids_dir/ids.txt"
fi
env ROOT="$ROOT" BFCL_PYTHON="$BFCL_PYTHON" \
  SGLANG_PYTHON=/home/liuyancheng/envs/sgl/bin/python \
  SGLANG_ROOT=/home/zhuyuhan/project/kvoffload-sglang-c2kv \
  MODEL_PATH=/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088 \
  TOKENIZER_PATH=/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507 \
  PYTHONPATH="$ROOT:/home/zhuyuhan/project/kvoffload-sglang-c2kv/python:/home/zhuyuhan/project/c2kv:/home/zhuyuhan/project/python_overlay${PYTHONPATH:+:$PYTHONPATH}" \
  BFCL_DATA_ROOT=/home/zhuyuhan/benchmarks/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
  METHODS="${METHODS:-streamingllm,h2o,snapkv_persistent,pyramidkv}" \
  MAX_EXAMPLES="$EXAMPLES" IDS_PATH="$IDS_PATH" REFERENCE_DETAILS_PATH=__NONE__ \
  DEVICES=6 PORTS="${PORTS:-35786}" RUN_ROOT="$RUN_ROOT" CLEAN_OUTPUT=0 \
  RUNTIME_HISTORY_KV_BACKEND=physical_eviction PERSISTENT_HISTORY_KV_SESSION=1 \
  HISTORY_KV_RETENTION_RATIO="${HISTORY_KV_RETENTION_RATIO:-0.312}" \
  MEM_FRACTION_STATIC=0.45 C2KV_POOL_FRACTION=0.05 \
  STRICT_RUNTIME_EVICTION=1 ALLOW_CLIENT_FALLBACK=0 \
  SGLANG_EXTRA_ARGS=--disable-cuda-graph \
  bash "$ROOT/c2kv_eval/scripts/run_bfcl_history_kv_baselines.sh"
