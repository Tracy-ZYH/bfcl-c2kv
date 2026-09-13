#!/usr/bin/env bash
set -euo pipefail

# Full tau2 airline (50 fixed tasks) compression-agnostic recovery matrix.
# Methods are sharded across two independent NPU servers; every method sees
# exactly the same task ids in the same order. Recovery is request-local only.

BFCL_ROOT="${BFCL_ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
TAU2_ROOT="${TAU2_ROOT:-/home/zhuyuhan/benchmarks/tau2}"
CHECKPOINT="${CHECKPOINT:-/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER="${TOKENIZER:-/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
TAU2_PYTHON="${TAU2_PYTHON:-/home/liuyancheng/envs/bench312/bin/python}"
DEVICES="${DEVICES:-5,6}"
PORTS="${PORTS:-35605,35606}"
MODEL="${MODEL:-c2kv-agent}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.05}"
UPSTREAM_TIMEOUT="${UPSTREAM_TIMEOUT:-600}"
# Existing completed arms peaked below 2.7k completion tokens.  A 4k ceiling
# is therefore non-binding for those results while stopping the observed H2O
# runaway (>18k tokens) before tau2's task timeout.
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
TAU2_TASK_IDS="${TAU2_TASK_IDS:-$(seq -s, 0 49)}"
EXPECTED_CASES="${EXPECTED_CASES:-50}"
METHOD_FILTER="${METHOD_FILTER:-all}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/runs/compression_agnostic_recovery_tau2_airline50_$(date +%Y%m%d_%H%M%S)}"

IFS=, read -r DEVICE_A DEVICE_B <<<"${DEVICES}"
IFS=, read -r PORT_A PORT_B <<<"${PORTS}"
if [[ -z "${DEVICE_A:-}" || -z "${DEVICE_B:-}" || -z "${PORT_A:-}" || -z "${PORT_B:-}" ]]; then
  echo "DEVICES and PORTS must each contain exactly two comma-separated values" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/manifests" "${RUN_ROOT}/full/tau2"

set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

SGLANG_IMPORT="$(env PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
  "${SGLANG_PYTHON}" -c 'import sglang; print(sglang.__file__)')"
TAU2_IMPORT="$(env PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}" \
  "${TAU2_PYTHON}" -c 'import tau2; print(tau2.__file__)')"
case "${SGLANG_IMPORT}" in "${SGLANG_ROOT}"/python/*) ;; *) echo "wrong SGLang import: ${SGLANG_IMPORT}" >&2; exit 1;; esac
case "${TAU2_IMPORT}" in "${TAU2_ROOT}"/src/tau2/*) ;; *) echo "wrong tau2 import: ${TAU2_IMPORT}" >&2; exit 1;; esac

METHODS=(
  c2kv4 c2kv4_replace_w2 c2kv4_append_w2
  streamingllm_r25 streamingllm_r25_replace_w2 streamingllm_r25_append_w2
  h2o_r25 snapkv_r25
  pyramidkv_r25 pyramidkv_r25_replace_w2 pyramidkv_r25_append_w2
)
ARMS=(
  c2kv4 c2kv4 c2kv4
  history_kv_streamingllm_r25 history_kv_streamingllm_r25 history_kv_streamingllm_r25
  history_kv_h2o_r25 history_kv_snapkv_persistent_r25
  history_kv_pyramidkv_r25 history_kv_pyramidkv_r25 history_kv_pyramidkv_r25
)
RECOVERY=(
  ''
  '{"operation":"replace","triggered":true,"selector":"first","window":2}'
  '{"operation":"append","triggered":true,"selector":"first","window":2}'
  ''
  '{"operation":"replace","triggered":true,"selector":"first","window":2}'
  '{"operation":"append","triggered":true,"selector":"first","window":2}'
  '' '' ''
  '{"operation":"replace","triggered":true,"selector":"first","window":2}'
  '{"operation":"append","triggered":true,"selector":"first","window":2}'
)

method_selected() {
  local method="$1"
  [[ "${METHOD_FILTER}" == "all" || ",${METHOD_FILTER}," == *",${method},"* ]]
}
SELECTED_COUNT=0
for method in "${METHODS[@]}"; do
  if method_selected "${method}"; then ((SELECTED_COUNT += 1)); fi
done
if (( SELECTED_COUNT == 0 )); then
  echo "METHOD_FILTER selected no known methods: ${METHOD_FILTER}" >&2
  exit 2
fi
printf '%s\n' "${METHODS[@]}" | while IFS= read -r method; do
  if method_selected "${method}"; then printf '%s\n' "${method}"; fi
done >"${RUN_ROOT}/manifests/selected_methods.txt"

cat >"${RUN_ROOT}/manifests/methods.tsv" <<'EOF'
method	portable_arm	recovery_mode
c2kv4	c2kv4	compression_only
c2kv4_replace_w2	c2kv4	replace_w2
c2kv4_append_w2	c2kv4	append_w2
streamingllm_r25	history_kv_streamingllm_r25	compression_only
streamingllm_r25_replace_w2	history_kv_streamingllm_r25	replace_w2
streamingllm_r25_append_w2	history_kv_streamingllm_r25	append_w2
h2o_r25	history_kv_h2o_r25	compression_only
snapkv_r25	history_kv_snapkv_persistent_r25	compression_only
pyramidkv_r25	history_kv_pyramidkv_r25	compression_only
pyramidkv_r25_replace_w2	history_kv_pyramidkv_r25	replace_w2
pyramidkv_r25_append_w2	history_kv_pyramidkv_r25	append_w2
EOF
cat >"${RUN_ROOT}/manifests/unsupported.tsv" <<'EOF'
method	portable_arm	compression_backend	recovery_mode	reason
h2o_r25_replace_w2	history_kv_h2o_r25	h2o	replace_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
h2o_r25_append_w2	history_kv_h2o_r25	h2o	append_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
snapkv_r25_replace_w2	history_kv_snapkv_persistent_r25	snapkv_persistent	replace_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
snapkv_r25_append_w2	history_kv_snapkv_persistent_r25	snapkv_persistent	append_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
EOF
{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "bfcl_commit=$(git -C "${BFCL_ROOT}" rev-parse HEAD)"
  echo "sglang_commit=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "c2kv_commit=$(git -C "${C2KV_ROOT}" rev-parse HEAD)"
  echo "sglang_import=${SGLANG_IMPORT}"
  echo "tau2_import=${TAU2_IMPORT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "tokenizer=${TOKENIZER}"
  echo "devices=${DEVICES} ports=${PORTS}"
  echo "task_set=airline expected_cases=${EXPECTED_CASES}"
  echo "task_ids=${TAU2_TASK_IDS}"
  echo "method_filter=${METHOD_FILTER} selected_count=${SELECTED_COUNT}"
  echo "upstream_timeout=${UPSTREAM_TIMEOUT} max_completion_tokens=${MAX_COMPLETION_TOKENS}"
  echo "trigger=controlled_always selector=first window=2 scope=request-local"
} >"${RUN_ROOT}/manifests/run_manifest.txt"
git -C "${BFCL_ROOT}" diff >"${RUN_ROOT}/manifests/bfcl_dirty.diff"
git -C "${SGLANG_ROOT}" diff >"${RUN_ROOT}/manifests/sglang_dirty.diff"

start_server() {
  local device="$1" port="$2" server_log="$3"
  (
    cd "${SGLANG_ROOT}"
    exec setsid env PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
      ASCEND_RT_VISIBLE_DEVICES="${device}" ASCEND_LAUNCH_BLOCKING=1 \
      TASK_QUEUE_ENABLE=1 SGLANG_DEBUG_MEMORY_POOL=1 \
      no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
      "${SGLANG_PYTHON}" -m sglang.launch_server \
        --model-path "${CHECKPOINT}" --tokenizer-path "${TOKENIZER}" \
        --served-model-name "${MODEL}" --model-impl sglang --device npu \
        --attention-backend ascend --tool-call-parser qwen25 --enable-c2kv \
        --dtype bfloat16 --c2kv-pool-fraction "${C2KV_POOL_FRACTION}" \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" --disable-cuda-graph \
        --host 127.0.0.1 --port "${port}"
  ) >"${server_log}" 2>&1 &
  STARTED_SERVER_PID=$!
}

stop_server() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then wait "${pid}" 2>/dev/null || true; return 0; fi
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${pid}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  else
    echo "refusing non-private process-group cleanup: pid=${pid} pgid=${pgid}" >&2
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  for attempt in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then wait "${pid}" 2>/dev/null || true; return 0; fi
    sleep 0.5
  done
  if [[ "${pgid}" == "${pid}" ]]; then kill -KILL -- "-${pgid}" 2>/dev/null || true
  else kill -KILL "${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}

wait_server() {
  local pid="$1" port="$2" log="$3"
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server exited before listening on ${port}; see ${log}" >&2
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "server timed out on ${port}; see ${log}" >&2
  return 1
}

run_shard() {
  local shard="$1" device="$2" server_port="$3" proxy_base="$4"
  local server_pid="" shard_status=0
  cleanup_server() { stop_server "${server_pid:-}"; }
  trap cleanup_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for index in "${!METHODS[@]}"; do
    if (( index % 2 != shard )); then continue; fi
    local method="${METHODS[$index]}" arm="${ARMS[$index]}" recovery="${RECOVERY[$index]}"
    if ! method_selected "${method}"; then continue; fi
    local server_log="${RUN_ROOT}/logs/server_tau2_${method}_card${device}_${server_port}.log"
    local out="${RUN_ROOT}/full/tau2/${method}" proxy_port=$((proxy_base + index))
    # One fresh server per arm: a failed/aborted history-KV request cannot
    # contaminate the next baseline's scheduler or page table.
    start_server "${device}" "${server_port}" "${server_log}"
    server_pid="${STARTED_SERVER_PID}"
    if ! wait_server "${server_pid}" "${server_port}" "${server_log}"; then
      echo "FAILED card=${device} tau2/${method}: server startup" >&2
      shard_status=1
      stop_server "${server_pid}"; server_pid=""
      continue
    fi
    local common=(
      -m c2kv_eval.portable.run --benchmark tau2 --arm "${arm}"
      --upstream "http://127.0.0.1:${server_port}"
      --user-upstream "http://127.0.0.1:${server_port}"
      --proxy-port "${proxy_port}" --proxy-python "${SGLANG_PYTHON}"
      --out "${out}" --exact-out --run-name "tau2_airline50_${method}_$(basename "${RUN_ROOT}")"
      --model "${MODEL}" --checkpoint "${CHECKPOINT}" --tokenizer "${TOKENIZER}"
      --reference-profile checkpoint-1088 --num-workers "${NUM_WORKERS}"
      --upstream-timeout "${UPSTREAM_TIMEOUT}"
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
      --bench-python "${TAU2_PYTHON}" --tau2-dir "${TAU2_ROOT}"
      --task-set airline --tau2-task-ids "${TAU2_TASK_IDS}"
      --tau2-num-trials 1 --tau2-max-steps 100 --tau2-timeout 1200
    )
    if [[ -n "${recovery}" ]]; then common+=(--recovery-control "${recovery}"); fi
    echo "RUN card=${device} tau2/${method}: arm=${arm} cases=${EXPECTED_CASES}"
    if ! env PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}" \
      "${TAU2_PYTHON}" "${common[@]}"; then
      echo "FAILED card=${device} tau2/${method}; continuing with later arms" >&2
      shard_status=1
    fi
    stop_server "${server_pid}"
    server_pid=""
  done
  trap - EXIT INT TERM
  return "${shard_status}"
}

PID_A="" PID_B=""
cleanup_workers() {
  local pid
  for pid in "${PID_A}" "${PID_B}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then kill -TERM "${pid}" 2>/dev/null || true; fi
  done
  for pid in "${PID_A}" "${PID_B}"; do
    if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi
  done
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_shard 0 "${DEVICE_A}" "${PORT_A}" 34750 & PID_A=$!
run_shard 1 "${DEVICE_B}" "${PORT_B}" 34850 & PID_B=$!
status=0
wait "${PID_A}" || status=1; PID_A=""
wait "${PID_B}" || status=1; PID_B=""
if [[ "${status}" -ne 0 ]]; then
  echo "at least one tau2 shard failed; inspect ${RUN_ROOT}/logs" >&2
  exit "${status}"
fi
trap - EXIT INT TERM

env PYTHONPATH="${BFCL_ROOT}" "${TAU2_PYTHON}" \
  "${BFCL_ROOT}/c2kv_eval/portable/ops/summarize_compression_agnostic_recovery.py" \
  --root "${RUN_ROOT}" --phase full --out "${RUN_ROOT}/summary.csv" \
  --expected-completed "${SELECTED_COUNT}" --expected-cases "${EXPECTED_CASES}" \
  --require-complete
touch "${RUN_ROOT}/FULL_COMPLETE"
echo "completed: ${RUN_ROOT}"
