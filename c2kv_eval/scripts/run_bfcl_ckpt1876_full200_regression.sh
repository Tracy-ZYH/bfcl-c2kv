#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
C2KV_MODEL_PATH="${C2KV_MODEL_PATH:-/home/zhuyuhan/project/c2kv/outputs/qwen3_4b_agent_c2kv_60k_248/sglang_ckpt1876}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL_PATH}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/qwen3_4b_agent_c2kv_60k_248_ckpt1876}"
IDS_PATH="${IDS_PATH:-${ROOT}/results/multi_turn_base_full200/ground_truth/episode_ids.txt}"
# Leave reference-details empty by default: the checked-in full_reference
# summary is turn-level oracle output, not the per-sample details schema that
# bfcl_history_kv_baselines indexes by row["id"].
REFERENCE_DETAILS_PATH="${REFERENCE_DETAILS_PATH:-__NONE__}"

DEVICES="${DEVICES:-4,5,6,7}"
PORTS="${PORTS:-34740,34750,34760,34770}"
IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
IFS=',' read -r -a PORT_LIST <<< "${PORTS}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"
SGLANG_DEVICE="${SGLANG_DEVICE:-cuda}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule --sampling-backend pytorch}"
SGLANG_DISABLE_TVM_FFI_JIT="${SGLANG_DISABLE_TVM_FFI_JIT:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.50}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.04}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"

RUN_FULL="${RUN_FULL:-1}"
RUN_R2="${RUN_R2:-1}"
RUN_R4="${RUN_R4:-1}"
RUN_R8="${RUN_R8:-1}"

run_one() {
  local label="$1"
  local model_path="$2"
  local methods="$3"
  local ratio="$4"
  local enable_c2kv="$5"
  local device="$6"
  local port="$7"
  local run_root="${OUTPUT_ROOT}/${label}"
  if [ "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ] && [ -e "${run_root}/run_manifest.json" ]; then
    echo "refusing to overwrite existing run output: ${run_root}" >&2
    echo "set ALLOW_EXISTING_OUTPUT=1 only if you intentionally want to rerun this cell" >&2
    exit 1
  fi
  mkdir -p "${run_root}"
  echo "[bfcl-ckpt1876] ${label}: device=${device} port=${port} model_path=${model_path} ratio=${ratio} enable_c2kv=${enable_c2kv}"
  ROOT="${ROOT}" \
  SGLANG_ROOT="${SGLANG_ROOT}" \
  BFCL_PYTHON="${BFCL_PYTHON}" \
  SGLANG_PYTHON="${SGLANG_PYTHON}" \
  MODEL_PATH="${model_path}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  MODEL_ID="${MODEL_ID}" \
  CATEGORY="multi_turn_base" \
  MAX_EXAMPLES="200" \
  IDS_PATH="${IDS_PATH}" \
  REFERENCE_DETAILS_PATH="${REFERENCE_DETAILS_PATH}" \
  METHODS="${methods}" \
  RATIO="${ratio}" \
  DEVICES="${device}" \
  PORTS="${port}" \
  RUN_ROOT="${run_root}" \
  CLEAN_OUTPUT="0" \
  RUN_COMPARE="0" \
  ASSERT_IDENTITY="0" \
  SGLANG_DEVICE="${SGLANG_DEVICE}" \
  SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND}" \
  SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS}" \
  SGLANG_DISABLE_TVM_FFI_JIT="${SGLANG_DISABLE_TVM_FFI_JIT}" \
  ENABLE_C2KV="${enable_c2kv}" \
  C2KV_GIST_TYPE="dynamic-interleave" \
  C2KV_GIST_PARAM="qkv" \
  C2KV_QUERY_PROJ="base" \
  C2KV_TOOLS_DUMP="full" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION}" \
  MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  bash "${ROOT}/c2kv_eval/scripts/run_bfcl_history_kv_baselines.sh"
}

if [ "${#DEVICE_LIST[@]}" -lt 4 ] || [ "${#PORT_LIST[@]}" -lt 4 ]; then
  echo "need four DEVICES and four PORTS for full/r2/r4/r8 parallel run" >&2
  exit 1
fi

if [ ! -s "${IDS_PATH}" ]; then
  echo "missing fixed full-200 ids: ${IDS_PATH}" >&2
  exit 1
fi
if [ ! -d "${BASE_MODEL_PATH}" ]; then
  echo "missing base model: ${BASE_MODEL_PATH}" >&2
  exit 1
fi
if [ "${RUN_R2}" = "1" ] || [ "${RUN_R4}" = "1" ] || [ "${RUN_R8}" = "1" ]; then
  if [ ! -s "${C2KV_MODEL_PATH}/c2kv_adapter.safetensors" ]; then
    echo "missing exported C2KV serving checkpoint: ${C2KV_MODEL_PATH}" >&2
    echo "run scripts/export_agent_c2kv_checkpoint_for_sglang.py first" >&2
    exit 1
  fi
fi

pids=()
if [ "${RUN_FULL}" = "1" ]; then
  run_one "full_200" "${BASE_MODEL_PATH}" "full" "4" "0" "${DEVICE_LIST[0]}" "${PORT_LIST[0]}" &
  pids+=("$!")
fi
if [ "${RUN_R2}" = "1" ]; then
  run_one "c2kv_r2_200" "${C2KV_MODEL_PATH}" "c2kv" "2" "1" "${DEVICE_LIST[1]}" "${PORT_LIST[1]}" &
  pids+=("$!")
fi
if [ "${RUN_R4}" = "1" ]; then
  run_one "c2kv_r4_200" "${C2KV_MODEL_PATH}" "c2kv" "4" "1" "${DEVICE_LIST[2]}" "${PORT_LIST[2]}" &
  pids+=("$!")
fi
if [ "${RUN_R8}" = "1" ]; then
  run_one "c2kv_r8_200" "${C2KV_MODEL_PATH}" "c2kv" "8" "1" "${DEVICE_LIST[3]}" "${PORT_LIST[3]}" &
  pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[bfcl-ckpt1876] outputs under ${OUTPUT_ROOT}"
exit "${status}"
