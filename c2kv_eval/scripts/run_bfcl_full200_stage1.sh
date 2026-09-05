#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/multi_turn_base_full200}"
DEVICE="${DEVICE:-7}"
PORT="${PORT:-34970}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"

mkdir -p "${OUTPUT_ROOT}/ground_truth" "${OUTPUT_ROOT}/full_reference" "${OUTPUT_ROOT}/summaries"

{
  echo "bfcl_git_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "sglang_git_commit=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "c2kv_git_commit=$(git -C "${C2KV_ROOT}" rev-parse HEAD)"
} >"${OUTPUT_ROOT}/commit_hashes.txt"

"${BFCL_PYTHON}" - "${OUTPUT_ROOT}/experiment_config.json" <<PY
import json, pathlib
path = pathlib.Path("${OUTPUT_ROOT}/experiment_config.json")
config = {
    "bfcl_git_commit": "$(git -C "${ROOT}" rev-parse HEAD)",
    "sglang_git_commit": "$(git -C "${SGLANG_ROOT}" rev-parse HEAD)",
    "c2kv_git_commit": "$(git -C "${C2KV_ROOT}" rev-parse HEAD)",
    "category": "${CATEGORY}",
    "max_examples": int("${MAX_EXAMPLES}"),
    "model_id": "${MODEL_ID}",
    "model_path": "${MODEL_PATH}",
    "tokenizer_path": "${TOKENIZER_PATH}",
    "ids_path": "",
    "reference_details_path": "",
    "temperature": 0.0,
    "stable52_filtering": False,
    "full_success_only": False,
}
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

METHODS=full \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS_PATH="__NONE__" \
DEVICES="${DEVICE}" \
PORTS="${PORT}" \
RUN_ROOT="${OUTPUT_ROOT}/full_reference" \
CLEAN_OUTPUT=1 \
RUN_COMPARE=1 \
MODEL_ID="${MODEL_ID}" \
MODEL_PATH="${MODEL_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_history_kv_baselines.sh"

"${BFCL_PYTHON}" -m c2kv_eval.analysis.bfcl_task_oracle \
  --category "${CATEGORY}" \
  --model "${MODEL_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-examples "${MAX_EXAMPLES}" \
  --require-full200 \
  --full-details-path "${OUTPUT_ROOT}/full_reference/full/logs/details.jsonl"

echo "Stage 1 complete: ${OUTPUT_ROOT}"
