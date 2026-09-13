#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/liuyancheng/envs/bench/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhuyuhan/runs/bfcl_full200_v5_fixedfull}"
FULL_REFERENCE_ROOT="${FULL_REFERENCE_ROOT:-/home/zhuyuhan/runs/bfcl_full200_v2/full_reference}"
FULL_REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS:-${FULL_REFERENCE_ROOT}/full/logs/details.jsonl}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507}"
DEVICES="${DEVICES:-5,6}"
PORTS="${PORTS:-35615,35616}"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
START_TIME="$(date +%s)"

export ROOT BFCL_PYTHON SGLANG_PYTHON SGLANG_ROOT C2KV_ROOT
export MODEL_ID MODEL_PATH DEVICES PORTS
export TOKENIZER_PATH
export PYTHONPATH="${SGLANG_ROOT}/python:${ROOT}:/home/zhuyuhan/project/python_overlay"
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---disable-cuda-graph}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
export C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.05}"

ROLLBACK_ROOT="${OUTPUT_ROOT}/recovery_reference_oracle/rollback"
D1_ROOT="${ROLLBACK_ROOT}/fixed_depth_i4_oracle_d1_kv_restore_strict"
REPAIR_ROOT="${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair"

echo "[1/4] Score completed Rollback D1"
mkdir -p "${D1_ROOT}/score" "${D1_ROOT}/logs"
(
  cd "${ROOT}"
  exec "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner \
    --model "${MODEL_ID}" \
    --test-category multi_turn_base \
    --result-dir "${D1_ROOT}/result" \
    --score-dir "${D1_ROOT}/score" \
    --partial-eval
) >"${D1_ROOT}/logs/eval_resume_${RUN_STAMP}.log" 2>&1

echo "[2/4] Resume Rollback D2 and run Rollback D4 on NPU 5,6"
VERIFIERS=oracle \
ROLLBACK_DEPTHS=4,2 \
ROLLBACK_POLICY=fixed_depth \
ROLLBACK_BACKEND=kv_restore_strict \
CATEGORY=multi_turn_base \
MAX_EXAMPLES=200 \
IDS_PATH=__NONE__ \
REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS}" \
COMPRESSION_RATIO=4 \
CHECKPOINT_INTERVAL=4 \
RUN_ROOT="${ROLLBACK_ROOT}" \
CLEAN_OUTPUT=0 \
RESUME=1 \
LOG_SUFFIX="_resume_${RUN_STAMP}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_fixed_depth_sweep.sh"

"${BFCL_PYTHON}" - "${ROLLBACK_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for depth in (1, 2, 4):
    mode = root / f"fixed_depth_i4_oracle_d{depth}_kv_restore_strict"
    paths = list((mode / "result").rglob("*_result.json"))
    if len(paths) != 1:
        raise SystemExit(f"Rollback D{depth}: expected one result file, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].open(encoding="utf-8") if line.strip()]
    ids = {row.get("id") for row in rows}
    if len(rows) != 200 or len(ids) != 200:
        raise SystemExit(f"Rollback D{depth}: incomplete rows={len(rows)} unique_ids={len(ids)}")
    print(f"Rollback D{depth}: complete rows=200 unique_ids=200")
PY

if find "${REPAIR_ROOT}" -type f -path '*/result/*_result.json' -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to mix with existing partial KV-repair results: ${REPAIR_ROOT}"
  echo "Inspect the existing arms and rerun only missing arms."
  exit 1
fi

echo "[3/4] Run Append/Replace arms on NPU 5,6"
ARMS=d_corr_w1,d_corr_w2,d_corr_w4,d_corr_w2_hint,d_corr_replace_w1,d_corr_replace_w2,d_corr_replace_w4 \
REPAIR_TRIGGER=oracle \
CATEGORY=multi_turn_base \
MAX_EXAMPLES=200 \
IDS_PATH=__NONE__ \
REFERENCE_DETAILS_PATH="${FULL_REFERENCE_DETAILS}" \
RATIO=4 \
CHECKPOINT_INTERVAL=4 \
RUN_ROOT="${REPAIR_ROOT}" \
CLEAN_OUTPUT=0 \
RUN_COMPARE=1 \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh"

echo "[4/4] Generate unified CSV/Markdown"
cd "${ROOT}"
"${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_full200_unified \
  --output-root "${OUTPUT_ROOT}" \
  --category multi_turn_base \
  --model "${MODEL_ID}" \
  --compression-run-root "${OUTPUT_ROOT}/compression_baselines" \
  "Full:full:${FULL_REFERENCE_ROOT}/full" \
  "C2KV:c2kv:${OUTPUT_ROOT}/compression_baselines/c2kv" \
  "StreamingLLM:streamingllm:${OUTPUT_ROOT}/compression_baselines/streamingllm" \
  "H2O:h2o:${OUTPUT_ROOT}/compression_baselines/h2o" \
  "SnapKV:snapkv_persistent:${OUTPUT_ROOT}/compression_baselines/snapkv_persistent" \
  "PyramidKV:pyramidkv:${OUTPUT_ROOT}/compression_baselines/pyramidkv" \
  "KIVI-QDQ:kivi:${OUTPUT_ROOT}/compression_baselines/kivi" \
  "Rollback D1:rollback_d1:${ROLLBACK_ROOT}/fixed_depth_i4_oracle_d1_kv_restore_strict" \
  "Rollback D2:rollback_d2:${ROLLBACK_ROOT}/fixed_depth_i4_oracle_d2_kv_restore_strict" \
  "Rollback D4:rollback_d4:${ROLLBACK_ROOT}/fixed_depth_i4_oracle_d4_kv_restore_strict" \
  "Replace W1:d_corr_replace_w1:${REPAIR_ROOT}/d_corr_replace_w1" \
  "Replace W2:d_corr_replace_w2:${REPAIR_ROOT}/d_corr_replace_w2" \
  "Replace W4:d_corr_replace_w4:${REPAIR_ROOT}/d_corr_replace_w4" \
  "Append W1:d_corr_w1:${REPAIR_ROOT}/d_corr_w1" \
  "Append W2:d_corr_w2:${REPAIR_ROOT}/d_corr_w2" \
  "Append W4:d_corr_w4:${REPAIR_ROOT}/d_corr_w4" \
  "Append-Hint:d_corr_w2_hint:${REPAIR_ROOT}/d_corr_w2_hint"

END_TIME="$(date +%s)"
printf 'resume_runtime_seconds=%s\n' "$((END_TIME - START_TIME))" \
  >"${OUTPUT_ROOT}/resume_total_runtime_${RUN_STAMP}.txt"
echo "Complete: ${OUTPUT_ROOT}/unified_full200.csv"
echo "Markdown: ${OUTPUT_ROOT}/summaries/unified_full200.md"
