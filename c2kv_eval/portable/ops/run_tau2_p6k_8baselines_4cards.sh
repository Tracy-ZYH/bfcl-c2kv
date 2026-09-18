#!/usr/bin/env bash
set -euo pipefail

# P6K portable tau2/BFCL baseline matrix.
# This script starts one SGLang server per method and runs available methods a
# wave at a time on DEVICES=4,5,6,7 by default. It delegates benchmark
# semantics to c2kv_eval.portable.run and tau2.

BFCL_ROOT="${BFCL_ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
TAU2_ROOT="${TAU2_ROOT:-/home/zhuyuhan/project/benchmarks/tau2-bench}"

SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
TAU2_PYTHON="${TAU2_PYTHON:-/home/zhuyuhan/miniconda3/envs/tau2-c2kv-gpu/bin/python}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
CUDA_HOME_FOR_JIT="${CUDA_HOME_FOR_JIT:-/usr/local/cuda-13.0}"

C2KV_MODEL_PATH="${C2KV_MODEL_PATH:-/home/zhuyuhan/project/c2kv/outputs/qwen3_4b_agent_c2kv_60k_248/sglang_ckpt1876}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-c2kv-agent}"
BENCHMARK="${BENCHMARK:-tau2}"
BFCL_IDS_PATH="${BFCL_IDS_PATH:-}"
BFCL_CATEGORY="${BFCL_CATEGORY:-multi_turn_base}"

TAU2_TASK_SET="${TAU2_TASK_SET:-airline}"
TAU2_TASK_IDS="${TAU2_TASK_IDS:-}"
TAU2_NUM_TRIALS="${TAU2_NUM_TRIALS:-1}"
MAX_TASKS="${MAX_TASKS:-}"
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-100}"
TAU2_TIMEOUT="${TAU2_TIMEOUT:-1800}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"

DEVICES_CSV="${DEVICES:-4,5,6,7}"
REQUESTED_DEVICES_CSV="${DEVICES_CSV}"
PORTS_CSV="${PORTS:-35740,35750,35760,35770}"
PROXY_PORTS_CSV="${PROXY_PORTS:-36740,36750,36760,36770}"
RESOLVE_DEVICE_UUIDS="${RESOLVE_DEVICE_UUIDS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.04}"
CACHEBLEND_C2KV_POOL_FRACTION="${CACHEBLEND_C2KV_POOL_FRACTION:-0.08}"
CACHEBLEND_ATTN_QUERY_CHUNK="${CACHEBLEND_ATTN_QUERY_CHUNK:-128}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-60000}"
RUN_ROOT="${RUN_ROOT:-${BFCL_ROOT}/results/tau2_p6k_ckpt1876_8baselines_$(date +%Y%m%d_%H%M%S)}"
REUSE_RESULTS_ROOT="${REUSE_RESULTS_ROOT:-}"
REUSE_METHODS="${REUSE_METHODS:-full,hiagent}"

IFS=',' read -r -a DEVICES_ARR <<<"${DEVICES_CSV}"
IFS=',' read -r -a PORTS_ARR <<<"${PORTS_CSV}"
IFS=',' read -r -a PROXY_PORTS_ARR <<<"${PROXY_PORTS_CSV}"

if [[ "${#DEVICES_ARR[@]}" -lt 1 || "${#PORTS_ARR[@]}" -ne "${#DEVICES_ARR[@]}" || "${#PROXY_PORTS_ARR[@]}" -ne "${#DEVICES_ARR[@]}" ]]; then
  echo "DEVICES, PORTS, and PROXY_PORTS must contain the same positive number of comma-separated values." >&2
  exit 2
fi
if [[ "${RESOLVE_DEVICE_UUIDS}" == "1" ]]; then
  for i in "${!DEVICES_ARR[@]}"; do
    if ! resolved_uuid="$(nvidia-smi --id="${DEVICES_ARR[${i}]}" \
      --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"; then
      echo "cannot resolve physical GPU ${DEVICES_ARR[${i}]} to UUID" >&2
      exit 2
    fi
    if [[ -z "${resolved_uuid}" ]]; then
      echo "empty UUID for physical GPU ${DEVICES_ARR[${i}]}" >&2
      exit 2
    fi
    DEVICES_ARR[${i}]="${resolved_uuid}"
  done
  DEVICES_CSV="$(IFS=,; echo "${DEVICES_ARR[*]}")"
fi
echo "GPU binding requested=${REQUESTED_DEVICES_CSV} resolved=${DEVICES_CSV}"
SLOT_COUNT="${#DEVICES_ARR[@]}"
for path in "${BFCL_ROOT}" "${SGLANG_ROOT}" "${C2KV_ROOT}" "${TAU2_ROOT}" "${C2KV_MODEL_PATH}" "${TOKENIZER_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done

case "${BENCHMARK}" in
  tau2) RUNNER_PYTHON="${TAU2_PYTHON}" ;;
  bfcl) RUNNER_PYTHON="${BFCL_PYTHON}" ;;
  *) echo "BENCHMARK must be tau2 or bfcl, got ${BENCHMARK}" >&2; exit 2 ;;
esac
if [[ ! -x "${RUNNER_PYTHON}" ]]; then
  echo "benchmark Python is not executable: ${RUNNER_PYTHON}" >&2
  exit 2
fi
if [[ "${BENCHMARK}" == "tau2" ]]; then
  if ! PYTHONPATH="${TAU2_ROOT}/src:${PYTHONPATH:-}" "${TAU2_PYTHON}" -c 'import tau2, loguru' >/dev/null 2>&1; then
    echo "tau2 dependencies are unavailable in ${TAU2_PYTHON}; expected imports: tau2, loguru" >&2
    exit 2
  fi
fi

for device in "${DEVICES_ARR[@]}"; do
  free_mb="$(nvidia-smi --id="${device}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d "[:space:]")"
  if [[ -n "${free_mb}" && "${free_mb}" -lt "${MIN_GPU_FREE_MB}" ]]; then
    echo "GPU ${device} has only ${free_mb} MiB free, below MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB}; refusing to start." >&2
    echo "Set MIN_GPU_FREE_MB=0 to bypass, or choose idle devices." >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/reference" "${RUN_ROOT}/manifests"
mkdir -p "${RUN_ROOT}/compare/${BENCHMARK}"

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "bfcl_commit=$(git -C "${BFCL_ROOT}" rev-parse HEAD)"
  echo "sglang_commit=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "tau2_root=${TAU2_ROOT}"
  echo "benchmark=${BENCHMARK}"
  echo "bfcl_category=${BFCL_CATEGORY}"
  echo "task_set=${TAU2_TASK_SET}"
  echo "task_ids=${TAU2_TASK_IDS}"
  echo "checkpoint=${C2KV_MODEL_PATH}"
  echo "tokenizer=${TOKENIZER_PATH}"
  echo "devices=${DEVICES_CSV}"
  echo "ports=${PORTS_CSV}"
  echo "proxy_ports=${PROXY_PORTS_CSV}"
  echo "c2kv_pool_fraction=${C2KV_POOL_FRACTION}"
  echo "cacheblend_c2kv_pool_fraction=${CACHEBLEND_C2KV_POOL_FRACTION}"
  echo "cacheblend_attn_query_chunk=${CACHEBLEND_ATTN_QUERY_CHUNK}"
  echo "reuse_results_root=${REUSE_RESULTS_ROOT}"
  echo "reuse_methods=${REUSE_METHODS}"
} >"${RUN_ROOT}/manifests/run_manifest.txt"

if [[ -n "${REUSE_RESULTS_ROOT}" ]]; then
  if [[ ! -d "${REUSE_RESULTS_ROOT}" ]]; then
    echo "reuse result root does not exist: ${REUSE_RESULTS_ROOT}" >&2
    exit 2
  fi
  IFS="," read -r -a REUSE_METHODS_ARR <<<"${REUSE_METHODS}"
  for reused_method in "${REUSE_METHODS_ARR[@]}"; do
    reused_source="${REUSE_RESULTS_ROOT}/compare/${BENCHMARK}/${reused_method}"
    reused_target="${RUN_ROOT}/compare/${BENCHMARK}/${reused_method}"
    if [[ ! -d "${reused_source}" ]]; then
      echo "missing reusable result: ${reused_source}" >&2
      exit 2
    fi
    if [[ ! -e "${reused_target}" ]]; then
      ln -s "${reused_source}" "${reused_target}"
    fi
  done
fi
if [[ -z "${REFERENCE_JSONL:-}" && -n "${REUSE_RESULTS_ROOT}" ]]; then
  REFERENCE_JSONL="${REUSE_RESULTS_ROOT}/reference/full_${BENCHMARK}_reference.jsonl"
else
  REFERENCE_JSONL="${REFERENCE_JSONL:-${RUN_ROOT}/reference/full_${BENCHMARK}_reference.jsonl}"
fi
SERVER_PIDS=()

cleanup_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_servers EXIT

wait_health() {
  local port="$1" log="$2"
  local attempt
  for attempt in $(seq 1 900); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if grep -qE "Scheduler hit an exception|unsupported GNU version|ninja exited with status|RuntimeError:|CUDA out of memory|Killed" "${log}" 2>/dev/null; then
      echo "SGLang failed while starting on port ${port}; tail follows: ${log}" >&2
      tail -n 160 "${log}" >&2 || true
      return 1
    fi
    sleep 1
    if (( attempt % 30 == 0 )); then
      echo "waiting for SGLang on port ${port} (${attempt}/900)"
    fi
  done
  echo "SGLang did not become healthy on port ${port}; tail follows: ${log}" >&2
  tail -n 160 "${log}" >&2 || true
  return 1
}

start_server() {
  local method="$1" device="$2" port="$3"
  local log="${RUN_ROOT}/logs/server_${method}_gpu${device}_port${port}.log"
  local persistent_args=()
  local pool_fraction="${C2KV_POOL_FRACTION}"
  case "${method}" in
    streamingllm_r25|h2o_r25|snapkv_r25|pyramidkv_r25|persistent_full_r100|joint_recent_trunc_r25|joint_snapkv_r25)
      persistent_args=(--enable-streaming-session --disable-radix-cache)
      ;;
    cacheblend)
      # CacheBlend stores the full raw history KV entry. Long tau2 sessions can
      # exceed the smaller compressor/recovery pool used by the other arms.
      pool_fraction="${CACHEBLEND_C2KV_POOL_FRACTION}"
      ;;
  esac
  (
    cd "${SGLANG_ROOT}"
    exec env \
      PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
      CUDA_VISIBLE_DEVICES="${device}" \
      CUDA_HOME="${CUDA_HOME_FOR_JIT}" \
      CUDA_PATH="${CUDA_HOME_FOR_JIT}" \
      PATH="${CUDA_HOME_FOR_JIT}/bin:${PATH}" \
      LD_LIBRARY_PATH="${CUDA_HOME_FOR_JIT}/lib64:${LD_LIBRARY_PATH:-}" \
      CC=/usr/bin/gcc-12 \
      CXX=/usr/bin/g++-12 \
      CUDAHOSTCXX=/usr/bin/g++-12 \
      NVCC_PREPEND_FLAGS=-ccbin=/usr/bin/g++-12 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      C2KV_CACHEBLEND_ATTN_QUERY_CHUNK="${CACHEBLEND_ATTN_QUERY_CHUNK}" \
      SGLANG_ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
      no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
      "${SGLANG_PYTHON}" -m sglang.launch_server \
        --model-path "${C2KV_MODEL_PATH}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --model-impl sglang \
        --device cuda \
        --attention-backend "${ATTENTION_BACKEND}" \
        --tool-call-parser qwen25 \
        --enable-c2kv \
        --c2kv-gist-type dynamic-interleave \
        --c2kv-gist-param qkv \
        --c2kv-query-proj base \
        --c2kv-tools-dump full \
        --dtype bfloat16 \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --c2kv-pool-fraction "${pool_fraction}" \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph \
        --disable-overlap-schedule \
        --sampling-backend pytorch \
        --host 127.0.0.1 \
        "${persistent_args[@]}" \
        --port "${port}"
  ) >"${log}" 2>&1 &
  SERVER_PIDS+=("$!")
  wait_health "${port}" "${log}"
}

stop_wave_servers() {
  cleanup_servers
  SERVER_PIDS=()
}

run_tau2_cell() {
  local method="$1" arm="$2" device="$3" server_port="$4" proxy_port="$5" extra_features="$6" reference_mode="$7"
  local out="${RUN_ROOT}/compare/${BENCHMARK}/${method}"
  local run_name="${BENCHMARK}_${method}_$(basename "${RUN_ROOT}")"
  local cmd=(
    -m c2kv_eval.portable.run
    --benchmark "${BENCHMARK}"
    --arm "${arm}"
    --backend sglang
    --upstream "http://127.0.0.1:${server_port}"
    --user-upstream "http://127.0.0.1:${server_port}"
    --proxy-port "${proxy_port}"
    --proxy-python "${SGLANG_PYTHON}"
    --out "${out}"
    --exact-out
    --run-name "${run_name}"
    --model "${SERVED_MODEL_NAME}"
    --checkpoint "${C2KV_MODEL_PATH}"
    --tokenizer "${TOKENIZER_PATH}"
    --num-workers "${NUM_WORKERS}"
    --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
  )
  case "${BENCHMARK}" in
    tau2)
      cmd+=(--bench-python "${TAU2_PYTHON}" --tau2-dir "${TAU2_ROOT}"
        --task-set "${TAU2_TASK_SET}" --tau2-num-trials "${TAU2_NUM_TRIALS}"
        --tau2-max-steps "${TAU2_MAX_STEPS}" --tau2-timeout "${TAU2_TIMEOUT}")
      [[ -n "${TAU2_TASK_IDS}" ]] && cmd+=(--tau2-task-ids "${TAU2_TASK_IDS}")
      [[ -n "${MAX_TASKS}" ]] && cmd+=(--max-tasks "${MAX_TASKS}")
      ;;
    bfcl)
      cmd+=(--bfcl-dir "${BFCL_ROOT}" --categories "${BFCL_CATEGORY}")
      if [[ -n "${MAX_TASKS}" ]]; then
        local run_ids
        if [[ -n "${BFCL_IDS_PATH}" && -s "${BFCL_IDS_PATH}" ]]; then
          run_ids="$(head -n "${MAX_TASKS}" "${BFCL_IDS_PATH}" | paste -sd, -)"
        else
          run_ids="$(PYTHONPATH="${BFCL_ROOT}:${PYTHONPATH:-}" "${BFCL_PYTHON}" -c 'import sys; from bfcl_eval.utils import load_dataset_entry; rows=load_dataset_entry(sys.argv[1], include_prereq=False, include_language_specific_hint=False); print(",".join(str(row["id"]) for row in rows[:int(sys.argv[2])]))' "${BFCL_CATEGORY}" "${MAX_TASKS}")"
        fi
        [[ -n "${run_ids}" ]] || { echo "no official BFCL IDs resolved for ${BFCL_CATEGORY}" >&2; return 2; }
        cmd+=(--run-ids "${run_ids}")
      fi
      ;;
    *) echo "BENCHMARK must be tau2 or bfcl, got ${BENCHMARK}" >&2; return 2 ;;
  esac
  if [[ -n "${extra_features}" ]]; then
    cmd+=(--capability-features "${extra_features}")
  fi
  case "${reference_mode}" in
    record) cmd+=(--record-reference "${REFERENCE_JSONL}") ;;
    use) cmd+=(--reference "${REFERENCE_JSONL}") ;;
    none) ;;
    *) echo "bad reference mode: ${reference_mode}" >&2; return 2 ;;
  esac

  echo "RUN method=${method} arm=${arm} gpu=${device} server=${server_port} proxy=${proxy_port}"
  env PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}:${SGLANG_ROOT}/python:${C2KV_ROOT}" \
    no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
    "${RUNNER_PYTHON}" "${cmd[@]}" >"${RUN_ROOT}/logs/run_${method}.log" 2>&1
}

run_wave() {
  local wave_name="$1"
  shift
  local specs=("$@")
  local i
  for i in "${!specs[@]}"; do
    IFS='|' read -r method arm features refmode <<<"${specs[$i]}"
    start_server "${method}" "${DEVICES_ARR[$i]}" "${PORTS_ARR[$i]}"
  done
  local pids=()
  for i in "${!specs[@]}"; do
    IFS='|' read -r method arm features refmode <<<"${specs[$i]}"
    run_tau2_cell "${method}" "${arm}" "${DEVICES_ARR[$i]}" "${PORTS_ARR[$i]}" "${PROXY_PORTS_ARR[$i]}" "${features}" "${refmode}" &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  stop_wave_servers
  if [[ "${rc}" -ne 0 ]]; then
    echo "wave ${wave_name} failed; see ${RUN_ROOT}/logs" >&2
    return "${rc}"
  fi
}

ALL_SPECS=(
  "full|full||record"
  "persistent_full_r100|history_kv_full_r100_persistent||none"
  "streamingllm_r25|history_kv_streamingllm_r25_persistent||none"
  "joint_recent_trunc_r25|joint_streamingllm_joint_r25||none"
  "h2o_r25|history_kv_h2o_r25_persistent||none"
  "snapkv_r25|history_kv_snapkv_persistent_r25_persistent||none"
  "joint_snapkv_r25|joint_snapkv_persistent_joint_r25||none"
  "pyramidkv_r25|history_kv_pyramidkv_r25_persistent||none"
  "recent_trunc_r25|recent_trunc_r25||none"
  "hiagent|hiagent_full|hiagent_trajectory_retrieval_v1|none"
  "acon|acon_hist_base||none"
  "cacheblend|cacheblend_r16|cacheblend_repair_extract_v1|none"
  "c2kv_r8_recover|c2kv_recover||use"
)

run_scheduled_batch() {
  local wave_name="$1"
  shift
  local batch=("$@")
  [[ "${#batch[@]}" -gt 0 ]] || return 0
  run_wave "${wave_name}" "${batch[@]}"
}

SELECTED_SPECS=()
if [[ -n "${RUN_METHODS:-}" ]]; then
  IFS="," read -r -a RUN_METHODS_ARR <<<"${RUN_METHODS}"
  for spec in "${ALL_SPECS[@]}"; do
    method="${spec%%|*}"
    for wanted in "${RUN_METHODS_ARR[@]}"; do
      if [[ "${method}" == "${wanted}" ]]; then
        SELECTED_SPECS+=("${spec}")
        break
      fi
    done
  done
else
  SELECTED_SPECS=("${ALL_SPECS[@]}")
fi

if [[ "${#SELECTED_SPECS[@]}" -eq 0 ]]; then
  echo "No methods selected. RUN_METHODS=${RUN_METHODS:-}" >&2
  exit 2
fi

index=0
wave_id=1
while [[ "${index}" -lt "${#SELECTED_SPECS[@]}" ]]; do
  batch=()
  slot=0
  while [[ "${slot}" -lt "${SLOT_COUNT}" && "${index}" -lt "${#SELECTED_SPECS[@]}" ]]; do
    next_spec="${SELECTED_SPECS[${index}]}"
    IFS='|' read -r next_method next_arm next_features next_refmode <<<"${next_spec}"
    if [[ "${next_refmode}" == "use" && ! -s "${REFERENCE_JSONL}" ]]; then
      # A record-reference arm earlier in this batch must finish before a
      # recovery arm starts. End this wave and retry after Full writes it.
      if [[ "${#batch[@]}" -gt 0 ]]; then
        break
      fi
      echo "${next_method} requires an existing full reference: ${REFERENCE_JSONL}" >&2
      echo "Include full in RUN_METHODS, or pass REFERENCE_JSONL/REUSE_RESULTS_ROOT from a completed Full run." >&2
      exit 1
    fi
    batch+=("${next_spec}")
    index=$((index + 1))
    slot=$((slot + 1))
  done
  run_scheduled_batch "wave${wave_id}" "${batch[@]}"
  wave_id=$((wave_id + 1))
done
for persistent_method in persistent_full_r100 streamingllm_r25 h2o_r25 snapkv_r25 pyramidkv_r25; do
  persistent_root="${RUN_ROOT}/compare/${BENCHMARK}/${persistent_method}"
  if [[ -d "${persistent_root}" ]]; then
    "${RUNNER_PYTHON}" \
      "${BFCL_ROOT}/c2kv_eval/portable/ops/validate_persistent_history_smoke.py" \
      --root "${persistent_root}" \
      $([[ "${persistent_method}" == "persistent_full_r100" ]] && printf '%s' '--expected-retention 1.0 --require-zero-eviction')
  fi
done
"${RUNNER_PYTHON}" "${BFCL_ROOT}/c2kv_eval/portable/ops/summarize_tau2_toolsandbox.py" \
  --root "${RUN_ROOT}" \
  --out "${RUN_ROOT}/${BENCHMARK}_paper_baselines_summary.csv"

echo "completed; results: ${RUN_ROOT}"
