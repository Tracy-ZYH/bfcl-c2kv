#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-52}"
RATIO="${RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
RECENT_FULL_UNITS="${RECENT_FULL_UNITS:-0}"
TEMPERATURE="${TEMPERATURE:-0}"
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS_PATH="${REFERENCE_DETAILS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"

DEVICES="${DEVICES:-1,2,3,4,5,6,7}"
PORTS="${PORTS:-34501,34502,34503,34504,34505,34506,34507}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_stable52_$(date +%Y%m%d_%H%M%S)}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.06}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
REPAIR_EXTRACT_SOURCE="${REPAIR_EXTRACT_SOURCE:-auto}"

METHODS="${METHODS:-full,c2kv,rollback_d1,rollback_d2,rollback_d4,replace_w1,replace_w2,replace_w4,replace_all,recompute_w2,append_w2,append_w2_hint,hint_only,append_masked_w2,sham_mech}"

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
IFS=',' read -r -a PORT_LIST <<< "${PORTS}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
ACTIVE_PIDS=()

if [ "${#DEVICE_LIST[@]}" -ne "${#PORT_LIST[@]}" ]; then
  echo "DEVICES and PORTS must have the same length."
  exit 1
fi

log_info() {
  echo "[$(date '+%F %T')] $*"
}

cleanup() {
  local status=$?
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

write_manifest() {
  mkdir -p "${RUN_ROOT}"
  local bfcl_commit
  local sglang_commit
  bfcl_commit="$(git -C "${ROOT}" rev-parse HEAD)"
  sglang_commit="$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  "${BFCL_PYTHON}" - "${RUN_ROOT}/run_manifest.json" <<PY
import json
import pathlib

manifest = {
    "bfcl_git_commit": "${bfcl_commit}",
    "sglang_git_commit": "${sglang_commit}",
    "model_path": "${MODEL_PATH}",
    "tokenizer_path": "${TOKENIZER_PATH}",
    "ids_path": "${IDS_PATH}",
    "reference_details_path": "${REFERENCE_DETAILS_PATH}",
    "category": "${CATEGORY}",
    "ratio": int("${RATIO}"),
    "checkpoint_interval": int("${CHECKPOINT_INTERVAL}"),
    "recent_full_units": int("${RECENT_FULL_UNITS}"),
    "temperature": float("${TEMPERATURE}"),
    "repair_extract_source": "${REPAIR_EXTRACT_SOURCE}",
    "c2kv_repair_extract_attn_impl": "${C2KV_REPAIR_EXTRACT_ATTN_IMPL:-prompt_flash}",
    "methods": "${METHODS}".split(","),
    "devices": "${DEVICES}",
    "ports": "${PORTS}",
}
path = pathlib.Path("${RUN_ROOT}/run_manifest.json")
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PY
}

repair_arm_for_method() {
  case "$1" in
    full) echo "full" ;;
    c2kv) echo "c2kv" ;;
    sham_mech) echo "d_sham_mech" ;;
    hint_only) echo "hint_only" ;;
    append_w2) echo "d_corr_w2" ;;
    append_w2_hint) echo "d_corr_w2_hint" ;;
    append_masked_w2) echo "append_masked_w2" ;;
    replace_w1) echo "d_corr_replace_w1" ;;
    replace_w2) echo "d_corr_replace_w2" ;;
    replace_w4) echo "d_corr_replace_w4" ;;
    replace_all) echo "d_corr_replace_all" ;;
    recompute_w2) echo "d_corr_recompute_w2" ;;
    *) echo "" ;;
  esac
}

run_repair_method() {
  local method="$1"
  local device="$2"
  local port="$3"
  local arm
  arm="$(repair_arm_for_method "${method}")"
  if [ -z "${arm}" ]; then
    echo "No repair arm mapping for method=${method}"
    exit 1
  fi
  log_info "[${method}] repair arm=${arm} device=${device} port=${port}"
  (
    cd "${ROOT}"
    ROOT="${ROOT}" \
    SGLANG_ROOT="${SGLANG_ROOT}" \
    MODEL_PATH="${MODEL_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    MODEL_ID="${MODEL_ID}" \
    CATEGORY="${CATEGORY}" \
    MAX_EXAMPLES="${MAX_EXAMPLES}" \
    RATIO="${RATIO}" \
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
    IDS_PATH="${IDS_PATH}" \
    REFERENCE_DETAILS_PATH="${REFERENCE_DETAILS_PATH}" \
    RUN_ROOT="${RUN_ROOT}" \
    DEVICES="${device}" \
    PORTS="${port}" \
    ARMS="${arm}" \
    CLEAN_OUTPUT=0 \
    RUN_COMPARE=0 \
    USE_REPAIR_PLAN=0 \
    REPAIR_TRIGGER=oracle \
    REPAIR_EXTRACT_SOURCE="${REPAIR_EXTRACT_SOURCE}" \
    C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION}" \
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS}" \
    bash c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh
  )
  if [ "${arm}" != "${method}" ]; then
    rm -rf "${RUN_ROOT:?}/${method}"
    mv "${RUN_ROOT}/${arm}" "${RUN_ROOT}/${method}"
  fi
}

run_rollback_method() {
  local method="$1"
  local device="$2"
  local port="$3"
  local depth="${method#rollback_d}"
  log_info "[${method}] rollback depth=${depth} device=${device} port=${port}"
  (
    cd "${ROOT}"
    MODEL_PATH="${MODEL_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    MODEL_ID="${MODEL_ID}" \
    CATEGORY="${CATEGORY}" \
    MAX_EXAMPLES="${MAX_EXAMPLES}" \
    COMPRESSION_RATIO="${RATIO}" \
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
    VERIFIER=oracle \
    RECOVERY_MODE=since_checkpoint \
    RECOVERY_HORIZON=suffix \
    ATTRIBUTION=whole_segment \
    ROLLBACK_POLICY=fixed_depth \
    ROLLBACK_DEPTH="${depth}" \
    ROLLBACK_BACKEND=kv_restore_strict \
    ENABLE_HICACHE=auto \
    DEVICE="${device}" \
    PORT="${port}" \
    IDS_PATH="${IDS_PATH}" \
    REFERENCE_DETAILS="${REFERENCE_DETAILS_PATH}" \
    RUN_ROOT="${RUN_ROOT}" \
    MODE="${method}" \
    CLEAN_OUTPUT=0 \
    RUN_COMPARE=0 \
    bash c2kv_eval/scripts/run_bfcl_history_multistep_checkpoint.sh
  )
}

run_method() {
  local method="$1"
  local device="$2"
  local port="$3"
  case "${method}" in
    rollback_d*) run_rollback_method "${method}" "${device}" "${port}" ;;
    *) run_repair_method "${method}" "${device}" "${port}" ;;
  esac
}

compare_unified() {
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_unified_recovery \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --methods "${METHODS}"
  )
}

log_info "Unified BFCL C2KV recovery comparison starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "METHODS=${METHODS}"
log_info "REPAIR_EXTRACT_SOURCE=${REPAIR_EXTRACT_SOURCE}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  rm -rf "${RUN_ROOT}"
fi
mkdir -p "${RUN_ROOT}/logs"
write_manifest > "${RUN_ROOT}/logs/manifest_write.log"

batch_pids=()
slot=0
for method in "${METHOD_LIST[@]}"; do
  device="${DEVICE_LIST[$slot]}"
  port="${PORT_LIST[$slot]}"
  (
    run_method "${method}" "${device}" "${port}"
  ) > "${RUN_ROOT}/logs/${method}.launcher.log" 2>&1 &
  batch_pids+=("$!")
  ACTIVE_PIDS+=("$!")
  slot=$((slot + 1))
  if [ "${slot}" -ge "${#DEVICE_LIST[@]}" ]; then
    for pid in "${batch_pids[@]}"; do
      wait "${pid}"
    done
    batch_pids=()
    slot=0
  fi
done
for pid in "${batch_pids[@]}"; do
  wait "${pid}"
done

compare_unified
log_info "Unified comparison done: ${RUN_ROOT}/unified_recovery_comparison.csv"
