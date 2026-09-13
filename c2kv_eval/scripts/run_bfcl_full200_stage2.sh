#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/multi_turn_base_full200}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"

RATIO="${RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
DEVICES="${DEVICES:-5,6,7}"
PORTS="${PORTS:-35050,35060,35070}"
COMPRESSION_METHODS="${COMPRESSION_METHODS:-c2kv,streamingllm,h2o,snapkv_persistent,pyramidkv,kivi}"
ROLLBACK_DEPTHS="${ROLLBACK_DEPTHS:-1,2,4}"
RECOVERY_ARMS="${RECOVERY_ARMS:-d_corr_w1,d_corr_w2,d_corr_w4,d_corr_w2_hint,d_corr_replace_w1,d_corr_replace_w2,d_corr_replace_w4}"
STAGE_CLEAN_OUTPUT="${STAGE_CLEAN_OUTPUT:-1}"

if [ "${CATEGORY}" != "multi_turn_base" ] || [ "${MAX_EXAMPLES}" != "200" ]; then
  echo "Full-200 stage requires CATEGORY=multi_turn_base and MAX_EXAMPLES=200; got CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES}."
  exit 1
fi

FIXED_FULL_REFERENCE_ROOT="${FIXED_FULL_REFERENCE_ROOT:-${OUTPUT_ROOT}/full_reference}"
FULL_REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS:-${FIXED_FULL_REFERENCE_ROOT}/full/logs/details.jsonl}"
if [ ! -s "${FULL_REFERENCE_DETAILS}" ]; then
  echo "Full reference details missing: ${FULL_REFERENCE_DETAILS}"
  exit 1
fi

"${BFCL_PYTHON}" - "${FULL_REFERENCE_DETAILS}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = []
with path.open(encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid Full reference JSON at line {line_no}: {exc}")
if len(rows) != 200:
    raise SystemExit(f"Full reference must contain exactly 200 episodes; found {len(rows)} in {path}")
ids = [str(row.get("id", row.get("question_id", ""))) for row in rows]
if any(not item for item in ids) or len(set(ids)) != 200:
    raise SystemExit("Full reference episode ids are missing or not unique")
summary_path = path.parent / "summary.json"
if not summary_path.exists():
    raise SystemExit(f"Full reference summary is missing: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary.get("num_examples") or 0) != 200 or int(summary.get("errors") or 0) != 0:
    raise SystemExit(
        "Full reference summary is incomplete: "
        f"num_examples={summary.get('num_examples')}, errors={summary.get('errors')}"
    )
print(f"Validated fixed Full reference: episodes={len(rows)}, unique_ids={len(set(ids))}")
PY

mkdir -p \
  "${OUTPUT_ROOT}/compression_baselines" \
  "${OUTPUT_ROOT}/recovery_reference_oracle" \
  "${OUTPUT_ROOT}/recovery_task_oracle" \
  "${OUTPUT_ROOT}/summaries"

"${BFCL_PYTHON}" - "${OUTPUT_ROOT}/experiment_config.json" <<PY
import json, pathlib
path = pathlib.Path("${OUTPUT_ROOT}/experiment_config.json")
config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
config.update({
    "bfcl_git_commit": "$(git -C "${ROOT}" rev-parse HEAD)",
    "sglang_git_commit": "$(git -C "${SGLANG_ROOT}" rev-parse HEAD)",
    "c2kv_git_commit": "$(git -C "${C2KV_ROOT}" rev-parse HEAD)",
    "model_id": "${MODEL_ID}",
    "model_path": "${MODEL_PATH}",
    "tokenizer_path": "${TOKENIZER_PATH}",
    "ids_path": "",
    "reference_details_path": "${FULL_REFERENCE_DETAILS}",
    "fixed_full_reference_root": "${FIXED_FULL_REFERENCE_ROOT}",
    "reuse_full_reference": True,
    "ratio": int("${RATIO}"),
    "checkpoint_interval": int("${CHECKPOINT_INTERVAL}"),
    "temperature": float("${TEMPERATURE}"),
    "stage2_recovery_trigger": "oracle_reference_drift",
})
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

METHODS="${COMPRESSION_METHODS}" \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS_PATH="${FULL_REFERENCE_DETAILS}" \
RATIO="${RATIO}" \
TEMPERATURE="${TEMPERATURE}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
RUN_ROOT="${OUTPUT_ROOT}/compression_baselines" \
CLEAN_OUTPUT="${STAGE_CLEAN_OUTPUT}" \
RUN_COMPARE=1 \
MODEL_ID="${MODEL_ID}" \
MODEL_PATH="${MODEL_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_history_kv_baselines.sh"

VERIFIERS=oracle \
ROLLBACK_DEPTHS="${ROLLBACK_DEPTHS}" \
ROLLBACK_POLICY=fixed_depth \
ROLLBACK_BACKEND=kv_restore_strict \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS}" \
COMPRESSION_RATIO="${RATIO}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
RUN_ROOT="${OUTPUT_ROOT}/recovery_reference_oracle/rollback" \
CLEAN_OUTPUT="${STAGE_CLEAN_OUTPUT}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_fixed_depth_sweep.sh"

ARMS="${RECOVERY_ARMS}" \
REPAIR_TRIGGER=oracle \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS_PATH="${FULL_REFERENCE_DETAILS}" \
RATIO="${RATIO}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
RUN_ROOT="${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair" \
CLEAN_OUTPUT="${STAGE_CLEAN_OUTPUT}" \
RUN_COMPARE=1 \
MODEL_ID="${MODEL_ID}" \
MODEL_PATH="${MODEL_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh"

"${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_full200_unified \
  --output-root "${OUTPUT_ROOT}" \
  --category "${CATEGORY}" \
  --model "${MODEL_ID}" \
  --compression-run-root "${OUTPUT_ROOT}/compression_baselines" \
  "Full:full:${FIXED_FULL_REFERENCE_ROOT}/full" \
  "C2KV:c2kv:${OUTPUT_ROOT}/compression_baselines/c2kv" \
  "StreamingLLM:streamingllm:${OUTPUT_ROOT}/compression_baselines/streamingllm" \
  "H2O:h2o:${OUTPUT_ROOT}/compression_baselines/h2o" \
  "SnapKV:snapkv_persistent:${OUTPUT_ROOT}/compression_baselines/snapkv_persistent" \
  "PyramidKV:pyramidkv:${OUTPUT_ROOT}/compression_baselines/pyramidkv" \
  "KIVI-QDQ:kivi:${OUTPUT_ROOT}/compression_baselines/kivi" \
  "Rollback D1:rollback_d1:${OUTPUT_ROOT}/recovery_reference_oracle/rollback/fixed_depth_i${CHECKPOINT_INTERVAL}_oracle_d1_kv_restore_strict" \
  "Rollback D2:rollback_d2:${OUTPUT_ROOT}/recovery_reference_oracle/rollback/fixed_depth_i${CHECKPOINT_INTERVAL}_oracle_d2_kv_restore_strict" \
  "Rollback D4:rollback_d4:${OUTPUT_ROOT}/recovery_reference_oracle/rollback/fixed_depth_i${CHECKPOINT_INTERVAL}_oracle_d4_kv_restore_strict" \
  "Replace W1:d_corr_replace_w1:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_w1" \
  "Replace W2:d_corr_replace_w2:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_w2" \
  "Replace W4:d_corr_replace_w4:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_w4" \
  "Append W1:d_corr_w1:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_w1" \
  "Append W2:d_corr_w2:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_w2" \
  "Append W4:d_corr_w4:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_w4" \
  "Append-Hint:d_corr_w2_hint:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_w2_hint"

echo "Stage 2 complete: ${OUTPUT_ROOT}/unified_full200.csv"
