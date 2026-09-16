#!/usr/bin/env bash
set -euo pipefail

# One official tau2 task and one official ToolSandbox scenario for each formal
# persistent 4x history-KV arm.  The validator rejects a run that merely gets
# a score but does not prove at least two turns reused one physical session.

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
DEVICE_TAU2="${DEVICE_TAU2:-5}"
DEVICE_TS="${DEVICE_TS:-6}"
PORT_TAU2="${PORT_TAU2:-35605}"
PORT_TS="${PORT_TS:-35606}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/runs/portable_persistent_history_smoke_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-c2kv-agent}"
TS_SCENARIO="${TS_SCENARIO:-send_message_with_contact_content_cellular_off}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.10}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tau2" "${RUN_ROOT}/toolsandbox"
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

ARMS=(
  history_kv_streamingllm_r25_persistent
  history_kv_h2o_r25_persistent
  history_kv_snapkv_persistent_r25_persistent
  history_kv_pyramidkv_r25_persistent
)
PIDS=()

start_server() {
  local device="$1" port="$2" name="$3"
  (
    cd "${SGLANG_ROOT}"
    exec setsid env \
      PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
      ASCEND_RT_VISIBLE_DEVICES="${device}" ASCEND_LAUNCH_BLOCKING=1 \
      TASK_QUEUE_ENABLE=1 SGLANG_DEBUG_MEMORY_POOL=1 \
      no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
      "${SGLANG_PYTHON}" -m sglang.launch_server \
        --model-path "${CHECKPOINT}" --tokenizer-path "${TOKENIZER}" \
        --served-model-name "${MODEL}" --model-impl sglang --device npu \
        --attention-backend ascend --tool-call-parser qwen25 --enable-c2kv \
        --enable-streaming-session --disable-radix-cache \
        --dtype bfloat16 --c2kv-pool-fraction "${C2KV_POOL_FRACTION}" \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" --disable-cuda-graph \
        --host 127.0.0.1 --port "${port}"
  ) >"${RUN_ROOT}/logs/server_${name}_card${device}_${port}.log" 2>&1 &
  STARTED_PID=$!
}

stop_server() {
  local pid="${1:-}" pgid
  [[ -n "${pid}" ]] || return 0
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${pid}" ]]; then kill -TERM -- "-${pgid}" 2>/dev/null || true
  else kill -TERM "${pid}" 2>/dev/null || true; fi
  for _ in $(seq 1 60); do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return; }
    sleep 0.5
  done
  if [[ "${pgid}" == "${pid}" ]]; then kill -KILL -- "-${pgid}" 2>/dev/null || true
  else kill -KILL "${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}

wait_server() {
  local pid="$1" port="$2"
  for _ in $(seq 1 900); do
    kill -0 "${pid}" 2>/dev/null || return 1
    curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

run_benchmark() {
  local benchmark="$1" device="$2" port="$3" bench_python="$4"
  local pid index=0
  start_server "${device}" "${port}" "${benchmark}"; pid="${STARTED_PID}"
  trap 'stop_server "${pid:-}"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  wait_server "${pid}" "${port}"
  for arm in "${ARMS[@]}"; do
    local out="${RUN_ROOT}/${benchmark}/${arm}" proxy_port=$((34700 + device * 20 + index))
    local common=(
      -m c2kv_eval.portable.run --benchmark "${benchmark}" --arm "${arm}"
      --upstream "http://127.0.0.1:${port}" --user-upstream "http://127.0.0.1:${port}"
      --proxy-port "${proxy_port}" --proxy-python "${SGLANG_PYTHON}"
      --out "${out}" --exact-out --run-name "persistent_smoke_${benchmark}_${arm}"
      --model "${MODEL}" --checkpoint "${CHECKPOINT}" --tokenizer "${TOKENIZER}"
      --reference-profile checkpoint-1088 --num-workers 1 --upstream-timeout 1200
    )
    if [[ "${benchmark}" == tau2 ]]; then
      env PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}" "${bench_python}" "${common[@]}" \
        --bench-python "${bench_python}" --tau2-dir "${TAU2_ROOT}" \
        --task-set airline --tau2-task-ids 3 --tau2-num-trials 1 \
        --tau2-max-steps 100 --tau2-timeout 1200
    else
      env PYTHONPATH="${BFCL_ROOT}:${TS_ROOT}" TS_PARALLEL=1 \
        TOOLSANDBOX_QWEN_TOOL_RESULT_COMPAT=1 \
        "${bench_python}" "${common[@]}" --bench-python "${bench_python}" \
        --toolsandbox-dir "${TS_ROOT}" --ts-scenarios "${TS_SCENARIO}"
    fi
    curl --noproxy '*' -fsS -X POST \
      "http://127.0.0.1:${port}/flush_cache?timeout=60" >/dev/null
    index=$((index + 1))
  done
  stop_server "${pid}"; pid=""
  trap - EXIT INT TERM
}

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
run_benchmark tau2 "${DEVICE_TAU2}" "${PORT_TAU2}" "${TAU2_PYTHON}" & PIDS+=("$!")
run_benchmark toolsandbox "${DEVICE_TS}" "${PORT_TS}" "${TS_PYTHON}" & PIDS+=("$!")
status=0
for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
PIDS=()
[[ "${status}" == 0 ]] || { echo "one or more smoke runners failed: ${RUN_ROOT}" >&2; exit 1; }

env PYTHONPATH="${BFCL_ROOT}" "${TAU2_PYTHON}" \
  "${BFCL_ROOT}/c2kv_eval/portable/ops/validate_persistent_history_smoke.py" \
  --root "${RUN_ROOT}"
touch "${RUN_ROOT}/PERSISTENT_SMOKE_COMPLETE"
echo "completed: ${RUN_ROOT}"
