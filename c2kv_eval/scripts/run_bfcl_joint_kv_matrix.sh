#!/usr/bin/env bash
# Run BFCL Tool/History scope arms against an already-running SGLang server.
# This script never starts or stops the model server.
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
BFCL_DIR="${BFCL_DIR:-/home/zhuyuhan/benchmarks/gorilla/berkeley-function-call-leaderboard}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/liuyancheng/envs/bench/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
CHECKPOINT="${CHECKPOINT:-/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER="${TOKENIZER:-/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507-FC}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:35605}"
CASE_SET="${CASE_SET:-smoke2}" # smoke2 | stable52 | full200
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/runs/bfcl_joint_kv_${CASE_SET}_$(date +%Y%m%d_%H%M%S)}"
PROXY_PORT_BASE="${PROXY_PORT_BASE:-34650}"
STABLE52_IDS="${STABLE52_IDS:-/home/zhuyuhan/recovery_zhuyuhan/fork192/inputs/correct_ids.txt}"
ARMS_CSV="${ARMS:-joint_h2o_tool_only_r25,joint_h2o_history_only_r25,joint_h2o_joint_r25,joint_snapkv_persistent_tool_only_r25,joint_snapkv_persistent_history_only_r25,joint_snapkv_persistent_joint_r25,joint_pyramidkv_tool_only_r25,joint_pyramidkv_history_only_r25,joint_pyramidkv_joint_r25,joint_cacheblend_history_only_r16}"

case "${CASE_SET}" in
  smoke2) RUN_IDS="multi_turn_base_5,multi_turn_base_12" ;;
  stable52)
    [[ -f "${STABLE52_IDS}" ]] || { echo "missing ${STABLE52_IDS}" >&2; exit 2; }
    RUN_IDS="$(paste -sd, "${STABLE52_IDS}")"
    [[ "$(wc -l < "${STABLE52_IDS}")" -eq 52 ]] || { echo "stable52 file is not 52 lines" >&2; exit 2; }
    ;;
  full200) RUN_IDS="" ;;
  *) echo "CASE_SET must be smoke2, stable52, or full200" >&2; exit 2 ;;
esac

[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing existing output ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "${RUN_ROOT}"
IFS=, read -r -a arms <<< "${ARMS_CSV}"

index=0
for arm in "${arms[@]}"; do
  out="${RUN_ROOT}/${arm}"
  args=(
    -m c2kv_eval.portable.run
    --benchmark bfcl --arm "${arm}"
    --upstream "${UPSTREAM}" --proxy-port "$((PROXY_PORT_BASE + index))"
    --proxy-python "${SGLANG_PYTHON}"
    --out "${out}" --exact-out --run-name "bfcl_joint_${CASE_SET}_${arm}"
    --model "${MODEL}" --checkpoint "${CHECKPOINT}" --tokenizer "${TOKENIZER}"
    --reference-profile checkpoint-1088
    --bfcl-dir "${BFCL_DIR}" --categories multi_turn_base
    --num-workers 1 --upstream-timeout 1200
    --capability-features cacheblend_repair_extract_v1
  )
  if [[ -n "${RUN_IDS}" ]]; then args+=(--run-ids "${RUN_IDS}"); fi
  env \
    PYTHONPATH="${ROOT}:${SGLANG_ROOT}/python:${C2KV_ROOT}:${BFCL_DIR}" \
    no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
    "${BFCL_PYTHON}" "${args[@]}" \
    2>&1 | tee "${RUN_ROOT}/${arm}.log"
  index=$((index + 1))
done

echo "completed ${RUN_ROOT}"
