#!/usr/bin/env bash
set -euo pipefail

# One SGLang server, sequential per-arm proxies.  PHASE=smoke_then_compare
# runs the fixed one-case gates first and starts the fixed ten-case comparison
# only if every official harness/scorer invocation succeeds and every smoke
# trajectory reaches a normal benchmark terminal state.

BFCL_ROOT="${BFCL_ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
TAU2_ROOT="${TAU2_ROOT:-/home/zhuyuhan/benchmarks/tau2}"
TS_ROOT="${TS_ROOT:-/home/zhuyuhan/benchmarks/ToolSandbox}"
CHECKPOINT="${CHECKPOINT:-/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER="${TOKENIZER:-/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
TAU2_PYTHON="${TAU2_PYTHON:-/home/liuyancheng/envs/bench312/bin/python}"
TS_PYTHON="${TS_PYTHON:-/home/liuyancheng/envs/benchts/bin/python}"
DEVICE="${DEVICE:-3}"
SERVER_PORT="${SERVER_PORT:-35603}"
PROXY_PORT_BASE="${PROXY_PORT_BASE:-34630}"
MODEL="${MODEL:-c2kv-agent}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.05}"
PHASE="${PHASE:-smoke_then_compare}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/runs/portable_tau2_toolsandbox_v1/$(date +%Y%m%d_%H%M%S)}"

case "${PHASE}" in
  smoke|compare|smoke_then_compare) ;;
  *) echo "PHASE must be smoke, compare, or smoke_then_compare" >&2; exit 2 ;;
esac

if [[ "${PHASE}" != compare && -e "${RUN_ROOT}" ]]; then
  echo "refusing to reuse output root ${RUN_ROOT}" >&2
  exit 1
fi
if [[ "${PHASE}" == compare && ! -f "${RUN_ROOT}/smoke/ALL_METHODS_PASSED" ]]; then
  echo "compare requires a successful smoke marker under the same RUN_ROOT" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/manifests"

set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

SGLANG_IMPORT="$({
  PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
    "${SGLANG_PYTHON}" -c 'import sglang; print(sglang.__file__)'
})"
echo "SGLang import: ${SGLANG_IMPORT}"
case "${SGLANG_IMPORT}" in
  "${SGLANG_ROOT}"/python/*) ;;
  *) echo "wrong SGLang import: ${SGLANG_IMPORT}" >&2; exit 1 ;;
esac

TAU2_IMPORT="$({
  PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}" \
    "${TAU2_PYTHON}" -c 'import tau2; print(tau2.__file__)'
})"
echo "tau2 import: ${TAU2_IMPORT}"
case "${TAU2_IMPORT}" in
  "${TAU2_ROOT}"/src/tau2/*) ;;
  *) echo "wrong tau2 import: ${TAU2_IMPORT}" >&2; exit 1 ;;
esac

PYTHONPATH="${TS_ROOT}" "${TS_PYTHON%/python}/tool_sandbox" --help \
  >"${RUN_ROOT}/manifests/toolsandbox_help_${PHASE}.txt" 2>&1

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "bfcl_commit=$(git -C "${BFCL_ROOT}" rev-parse HEAD)"
  echo "sglang_commit=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "c2kv_commit=$(git -C "${C2KV_ROOT}" rev-parse HEAD)"
  echo "sglang_python=${SGLANG_PYTHON}"
  echo "tau2_python=${TAU2_PYTHON}"
  echo "toolsandbox_python=${TS_PYTHON}"
  echo "sglang_import=${SGLANG_IMPORT}"
  echo "tau2_import=${TAU2_IMPORT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "tokenizer=${TOKENIZER}"
  echo "reference_profile=checkpoint-1088"
  echo "npu_device=${DEVICE}"
  echo "endpoint=http://127.0.0.1:${SERVER_PORT}"
} >"${RUN_ROOT}/manifests/run_manifest_${PHASE}.txt"
git -C "${SGLANG_ROOT}" status --short \
  >"${RUN_ROOT}/manifests/sglang_status_${PHASE}.txt"
git -C "${SGLANG_ROOT}" diff \
  >"${RUN_ROOT}/manifests/sglang_dirty_${PHASE}.diff"

if [[ ! -e "${RUN_ROOT}/manifests/methods.tsv" ]]; then
cat >"${RUN_ROOT}/manifests/methods.tsv" <<'EOF'
method	portable_arm	recovery_control	scope
full	full		
c2kv4	c2kv4		
c2kv8	c2kv		
append_first	c2kv4	{"operation":"append","triggered":true,"selector":"first"}	request-local retry; no environment rollback
replace_first	c2kv4	{"operation":"replace","triggered":true,"selector":"first"}	request-local retry; no environment rollback
full_kv_retry	c2kv4	{"operation":"retry_full","triggered":true}	Full-KV request-local retry; no environment rollback
streamingllm_r312	history_kv_streamingllm_r312		portable history-boundary adaptation
h2o_r312	history_kv_h2o_r312		portable history-boundary adaptation
snapkv_r312	history_kv_snapkv_r312		portable persistent history-boundary adaptation
pyramidkv_r312	history_kv_pyramidkv_r312		portable shared-page-table globalized adaptation
EOF
fi

SERVER_PID=""
cleanup() {
  [[ -n "${SERVER_PID}" ]] || return 0
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" 2>/dev/null || true
    return 0
  fi
  local pgid
  pgid="$(ps -o pgid= -p "${SERVER_PID}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${SERVER_PID}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  else
    echo "refusing non-private process-group cleanup: pid=${SERVER_PID} pgid=${pgid}" >&2
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
  fi
  for attempt in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      return 0
    fi
    sleep 0.5
  done
  if [[ "${pgid}" == "${SERVER_PID}" ]]; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  else
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
  cd "${SGLANG_ROOT}"
  exec setsid env \
    PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
    ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
    ASCEND_LAUNCH_BLOCKING=1 TASK_QUEUE_ENABLE=1 \
    SGLANG_DEBUG_MEMORY_POOL=1 \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_EMPTY_CACHE_INTERVAL=1 \
    no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
    "${SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${CHECKPOINT}" \
      --tokenizer-path "${TOKENIZER}" \
      --served-model-name "${MODEL}" \
      --model-impl sglang --device npu --attention-backend ascend \
      --tool-call-parser qwen25 --enable-c2kv --dtype bfloat16 \
      --c2kv-pool-fraction "${C2KV_POOL_FRACTION}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --disable-cuda-graph --host 127.0.0.1 --port "${SERVER_PORT}"
) >"${RUN_ROOT}/logs/server_${PHASE}_card${DEVICE}_${SERVER_PORT}.log" 2>&1 &
SERVER_PID="$!"

for attempt in $(seq 1 900); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 160 "${RUN_ROOT}/logs/server_${PHASE}_card${DEVICE}_${SERVER_PORT}.log" >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" \
      >/dev/null 2>&1; then
    break
  fi
  if (( attempt % 30 == 0 )); then echo "waiting for SGLang (${attempt}/900)"; fi
  sleep 1
done
curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null

METHOD_KEYS=(full c2kv4 c2kv8 append_first replace_first full_kv_retry streamingllm_r312 h2o_r312 snapkv_r312 pyramidkv_r312)
ARMS=(full c2kv4 c2kv c2kv4 c2kv4 c2kv4 history_kv_streamingllm_r312 history_kv_h2o_r312 history_kv_snapkv_r312 history_kv_pyramidkv_r312)
RECOVERY=(
  '' '' ''
  '{"operation":"append","triggered":true,"selector":"first"}'
  '{"operation":"replace","triggered":true,"selector":"first"}'
  '{"operation":"retry_full","triggered":true}'
  '' '' '' ''
)
# Task 3 is the shortest/highest-normal-termination airline case in the
# existing local tau2 runs (31/35 normal); task 0 was only 34/59 and made a
# poor infrastructure smoke because model loops dominated the signal.
TAU2_SMOKE_IDS="3"
TAU2_COMPARE_IDS="0,1,3,4,5,7,9,10,11,12"
# Use the information-complete multi-tool variant for infrastructure smoke.
# The *_multiple_user_turn variant is useful for model-quality evaluation but
# Qwen's local user simulator can loop until ToolSandbox's 30-message cap,
# which confounds harness/recovery validation with simulator quality.
TS_SMOKE_IDS="send_message_with_contact_content_cellular_off"
TS_COMPARE_IDS="search_message_with_recency_latest_multiple_user_turn,search_message_with_recency_oldest_multiple_user_turn,modify_contact_with_message_recency_multiple_user_turn,remove_contact_by_phone_multiple_user_turn,send_message_with_contact_content_cellular_off_multiple_user_turn,update_contact_relationship_with_relationship_multiple_user_turn,find_days_till_holiday_multiple_user_turn,find_temperature_f_with_location_and_time_diff_multiple_user_turn,find_distance_with_location_name_multiple_user_turn,add_reminder_content_and_date_and_time_multiple_user_turn"
PORT_OFFSET=0

run_one() {
  local phase="$1" benchmark="$2" method="$3" arm="$4" recovery="$5" ids="$6"
  local out="${RUN_ROOT}/${phase}/${benchmark}/${method}"
  local proxy_port=$((PROXY_PORT_BASE + PORT_OFFSET))
  PORT_OFFSET=$((PORT_OFFSET + 1))
  if [[ -e "${out}" ]]; then
    echo "refusing to overwrite ${out}" >&2
    return 1
  fi
  curl --noproxy '*' -fsS -X POST \
    "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=60" >/dev/null
  local common=(
    -m c2kv_eval.portable.run --benchmark "${benchmark}" --arm "${arm}"
    --upstream "http://127.0.0.1:${SERVER_PORT}"
    --user-upstream "http://127.0.0.1:${SERVER_PORT}"
    --proxy-port "${proxy_port}" --proxy-python "${SGLANG_PYTHON}"
    --out "${out}" --exact-out --run-name "${phase}_${benchmark}_${method}_$(basename "${RUN_ROOT}")"
    --model "${MODEL}" --checkpoint "${CHECKPOINT}" --tokenizer "${TOKENIZER}"
    --reference-profile checkpoint-1088 --num-workers 1
  )
  if [[ -n "${recovery}" ]]; then common+=(--recovery-control "${recovery}"); fi
  echo "RUN ${phase}/${benchmark}/${method}: arm=${arm} proxy=${proxy_port}"
  if [[ "${benchmark}" == tau2 ]]; then
    env PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}" \
      "${TAU2_PYTHON}" "${common[@]}" \
      --bench-python "${TAU2_PYTHON}" --tau2-dir "${TAU2_ROOT}" \
      --task-set airline --tau2-task-ids "${ids}" --tau2-num-trials 1 \
      --tau2-max-steps 100 --tau2-timeout 1200
  else
    env PYTHONPATH="${BFCL_ROOT}:${TS_ROOT}" TS_PARALLEL=1 \
      "${TS_PYTHON}" "${common[@]}" \
      --bench-python "${TS_PYTHON}" --toolsandbox-dir "${TS_ROOT}" \
      --ts-scenarios "${ids}"
  fi
}

run_phase() {
  local phase="$1"
  local tau_ids ts_ids
  if [[ "${phase}" == smoke ]]; then
    tau_ids="${TAU2_SMOKE_IDS}"; ts_ids="${TS_SMOKE_IDS}"
  else
    tau_ids="${TAU2_COMPARE_IDS}"; ts_ids="${TS_COMPARE_IDS}"
  fi
  for index in "${!METHOD_KEYS[@]}"; do
    run_one "${phase}" tau2 "${METHOD_KEYS[$index]}" "${ARMS[$index]}" \
      "${RECOVERY[$index]}" "${tau_ids}"
    run_one "${phase}" toolsandbox "${METHOD_KEYS[$index]}" "${ARMS[$index]}" \
      "${RECOVERY[$index]}" "${ts_ids}"
  done
  touch "${RUN_ROOT}/${phase}/ALL_METHODS_SCORED"
  local summarize=(
    "${TAU2_PYTHON}"
    "${BFCL_ROOT}/c2kv_eval/portable/ops/summarize_tau2_toolsandbox.py"
    --root "${RUN_ROOT}" --out "${RUN_ROOT}/summary_through_${phase}.csv"
  )
  if [[ "${phase}" == smoke ]]; then
    if ! "${summarize[@]}" --require-normal-termination; then
      touch "${RUN_ROOT}/${phase}/SMOKE_QUALITY_GATE_FAILED"
      echo "smoke was scored but did not pass the normal-termination gate" >&2
      return 1
    fi
  else
    "${summarize[@]}"
  fi
  touch "${RUN_ROOT}/${phase}/ALL_METHODS_PASSED"
}

if [[ "${PHASE}" == smoke || "${PHASE}" == smoke_then_compare ]]; then
  run_phase smoke
fi
if [[ "${PHASE}" == compare || "${PHASE}" == smoke_then_compare ]]; then
  run_phase compare
fi

echo "completed ${PHASE}; results: ${RUN_ROOT}"
