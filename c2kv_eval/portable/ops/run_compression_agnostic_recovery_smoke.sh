#!/usr/bin/env bash
set -euo pipefail

# Two independent one-case official-harness smoke matrices.  tau2 runs on
# DEVICE_TAU2 (default card 5), ToolSandbox on DEVICE_TS (default card 6).
# Every recovery is a controlled always-triggered request-local W2 retry; it
# never rolls back an external tool/environment transaction.

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
RUN_TAU2="${RUN_TAU2:-1}"
RUN_TOOLSANDBOX="${RUN_TOOLSANDBOX:-1}"
PORT_TAU2="${PORT_TAU2:-35605}"
PORT_TS="${PORT_TS:-35606}"
PROXY_BASE_TAU2="${PROXY_BASE_TAU2:-34750}"
# Keep ToolSandbox disjoint from the tau2 full runner (34750/34850 ranges).
PROXY_BASE_TS="${PROXY_BASE_TS:-34950}"
MODEL="${MODEL:-c2kv-agent}"
# Use the shorter information-complete multi-tool case.  The native
# *_multiple_user_turn test spends much of the fixed 30-message budget on
# progressive disclosure and often reaches the cap immediately after a
# successful tool result.  Qwen's opaque-result ambiguity is handled by the
# explicit opt-in tool-name compatibility layer below.
TS_SCENARIO="${TS_SCENARIO:-send_message_with_contact_content_cellular_off}"
TS_CANONICAL="${TS_CANONICAL:-0}"
TS_PARALLEL="${TS_PARALLEL:-1}"
# Keep the combined explicit KV reservation at 0.50, but give the persistent
# history pool enough headroom for a multi-case ToolSandbox arm.  With the old
# 0.45 + 0.05 split, ten long scenarios reproducibly drove the C2KV pool to
# 99--100% even though the ordinary request KV pool was almost empty.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.10}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/runs/compression_agnostic_recovery_4x_smoke_$(date +%Y%m%d_%H%M%S)}"
EXPECTED_CASES="${EXPECTED_CASES:-1}"
MAX_C2KV_POOL_USAGE="${MAX_C2KV_POOL_USAGE:-0.90}"
RESUME="${RESUME:-0}"
PHASE="${PHASE:-smoke}"
case "${PHASE}" in
  smoke) MARKER_PREFIX=SMOKE ;;
  canonical) MARKER_PREFIX=CANONICAL ;;
  *) echo "unsupported PHASE=${PHASE} (expected smoke or canonical)" >&2; exit 1 ;;
esac

if [[ -e "${RUN_ROOT}" && "${RESUME}" != 1 ]]; then
  echo "refusing to overwrite ${RUN_ROOT}" >&2
  exit 1
fi
if [[ "${RUN_TAU2}" != 1 && "${RUN_TOOLSANDBOX}" != 1 ]]; then
  echo "at least one of RUN_TAU2/RUN_TOOLSANDBOX must be 1" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/manifests" "${RUN_ROOT}/${PHASE}"

set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

SGLANG_IMPORT="$(env PYTHONPATH="${SGLANG_ROOT}/python:${BFCL_ROOT}:${C2KV_ROOT}" \
  "${SGLANG_PYTHON}" -c 'import sglang; print(sglang.__file__)')"
case "${SGLANG_IMPORT}" in
  "${SGLANG_ROOT}"/python/*) ;;
  *) echo "wrong SGLang import: ${SGLANG_IMPORT}" >&2; exit 1 ;;
esac
TS_IMPORT=""
if [[ "${RUN_TOOLSANDBOX}" == 1 ]]; then
  TS_IMPORT="$(env PYTHONPATH="${TS_ROOT}:${BFCL_ROOT}" "${TS_PYTHON}" -c \
    'import tool_sandbox; from tool_sandbox.roles.openai_api_agent import _annotate_tool_results_for_qwen; print(tool_sandbox.__file__)')"
  case "${TS_IMPORT}" in
    "${TS_ROOT}"/tool_sandbox/*) ;;
    *) echo "wrong ToolSandbox import or missing Qwen compatibility patch: ${TS_IMPORT}" >&2; exit 1 ;;
  esac
fi
if [[ "${RUN_TOOLSANDBOX}" == 1 && "${TS_CANONICAL}" == 1 ]]; then
  # Resolve the canonical set from the exact ToolSandbox checkout instead of
  # maintaining a second hand-written list.  NO_DISTRACTION_TOOLS selects one
  # base scenario per task and excludes all distraction/scrambled variants.
  TS_SCENARIO="$(env PYTHONPATH="${TS_ROOT}:${BFCL_ROOT}" "${TS_PYTHON}" -c '
from tool_sandbox.common.scenario import ScenarioCategories
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios
scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
print(",".join(name for name, scenario in scenarios.items()
               if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories))
')"
  EXPECTED_CASES="$(awk -F, '{print NF}' <<<"${TS_SCENARIO}")"
  if [[ "${EXPECTED_CASES}" -ne 129 ]]; then
    echo "expected 129 canonical ToolSandbox scenarios, found ${EXPECTED_CASES}" >&2
    exit 1
  fi
fi

cat >"${RUN_ROOT}/manifests/unsupported.tsv" <<'EOF'
method	portable_arm	compression_backend	recovery_mode	reason
h2o_r25_replace_w2	history_kv_h2o_r25	h2o	replace_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
h2o_r25_append_w2	history_kv_h2o_r25	h2o	append_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
snapkv_r25_replace_w2	history_kv_snapkv_persistent_r25	snapkv_persistent	replace_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
snapkv_r25_append_w2	history_kv_snapkv_persistent_r25	snapkv_persistent	append_w2	headwise source indices cannot be exactly deduplicated in the current shared dense-slot entry
EOF
cat >"${RUN_ROOT}/manifests/methods.tsv" <<'EOF'
method	portable_arm	recovery_control	status
c2kv4	c2kv4		run
c2kv4_replace_w2	c2kv4	{"operation":"replace","triggered":true,"selector":"first","window":2}	run
c2kv4_append_w2	c2kv4	{"operation":"append","triggered":true,"selector":"first","window":2}	run
streamingllm_r25	history_kv_streamingllm_r25		run
streamingllm_r25_replace_w2	history_kv_streamingllm_r25	{"operation":"replace","triggered":true,"selector":"first","window":2}	run
streamingllm_r25_append_w2	history_kv_streamingllm_r25	{"operation":"append","triggered":true,"selector":"first","window":2}	run
h2o_r25	history_kv_h2o_r25		run
h2o_r25_replace_w2	history_kv_h2o_r25	{"operation":"replace","triggered":true,"selector":"first","window":2}	unsupported_headwise_dedup
h2o_r25_append_w2	history_kv_h2o_r25	{"operation":"append","triggered":true,"selector":"first","window":2}	unsupported_headwise_dedup
snapkv_r25	history_kv_snapkv_persistent_r25		run
snapkv_r25_replace_w2	history_kv_snapkv_persistent_r25	{"operation":"replace","triggered":true,"selector":"first","window":2}	unsupported_headwise_dedup
snapkv_r25_append_w2	history_kv_snapkv_persistent_r25	{"operation":"append","triggered":true,"selector":"first","window":2}	unsupported_headwise_dedup
pyramidkv_r25	history_kv_pyramidkv_r25		run
pyramidkv_r25_replace_w2	history_kv_pyramidkv_r25	{"operation":"replace","triggered":true,"selector":"first","window":2}	run
pyramidkv_r25_append_w2	history_kv_pyramidkv_r25	{"operation":"append","triggered":true,"selector":"first","window":2}	run
EOF

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "bfcl_commit=$(git -C "${BFCL_ROOT}" rev-parse HEAD)"
  echo "sglang_commit=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "c2kv_commit=$(git -C "${C2KV_ROOT}" rev-parse HEAD)"
  echo "sglang_import=${SGLANG_IMPORT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "tokenizer=${TOKENIZER}"
  echo "tau2_python=${TAU2_PYTHON}"
  echo "toolsandbox_python=${TS_PYTHON}"
  echo "toolsandbox_import=${TS_IMPORT}"
  echo "tau2_device=${DEVICE_TAU2} endpoint=http://127.0.0.1:${PORT_TAU2}"
  echo "toolsandbox_device=${DEVICE_TS} endpoint=http://127.0.0.1:${PORT_TS}"
  echo "tau2_proxy_base=${PROXY_BASE_TAU2}"
  echo "toolsandbox_proxy_base=${PROXY_BASE_TS}"
  echo "toolsandbox_scenario=${TS_SCENARIO}"
  echo "toolsandbox_canonical=${TS_CANONICAL}"
  echo "toolsandbox_parallel=${TS_PARALLEL}"
  echo "toolsandbox_qwen_tool_result_compat=1"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "c2kv_pool_fraction=${C2KV_POOL_FRACTION}"
  echo "trigger=controlled_always selector=first window=2 scope=request-local"
} >"${RUN_ROOT}/manifests/run_manifest.txt"
if [[ "${RUN_TOOLSANDBOX}" == 1 ]]; then
  tr ',' '\n' <<<"${TS_SCENARIO}" >"${RUN_ROOT}/manifests/toolsandbox_scenarios.txt"
fi
git -C "${BFCL_ROOT}" diff >"${RUN_ROOT}/manifests/bfcl_dirty.diff"
git -C "${SGLANG_ROOT}" diff >"${RUN_ROOT}/manifests/sglang_dirty.diff"

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

start_server() {
  local device="$1" port="$2" benchmark="$3"
  (
    cd "${SGLANG_ROOT}"
    # Give this server and every multiprocessing child a private process
    # group.  Killing only launch_server leaves the scheduler holding NPU
    # memory; a private group lets cleanup target both without touching tmux.
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
  ) >"${RUN_ROOT}/logs/server_${benchmark}_card${device}_${port}.log" 2>&1 &
  STARTED_SERVER_PID=$!
}

stop_server() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi
  # start_server uses setsid, so PGID == launcher PID and this group contains
  # only this smoke's launcher/workers.  Refuse a group signal if that
  # invariant is not true.
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" != "${pid}" ]]; then
    echo "refusing non-private process-group cleanup: pid=${pid} pgid=${pgid}" >&2
    kill -TERM "${pid}" 2>/dev/null || true
  else
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  fi
  for attempt in $(seq 1 60); do
    if [[ "${pgid}" == "${pid}" ]]; then
      if ! kill -0 -- "-${pgid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        return 0
      fi
    elif ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.5
  done
  if [[ "${pgid}" == "${pid}" ]]; then
    echo "server group ${pgid} ignored TERM; sending KILL" >&2
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  else
    echo "server ${pid} ignored TERM; sending KILL to launcher only" >&2
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  # Do not return while a scheduler child from this private group can still
  # retain NPU memory.  The group was validated above, so this never targets
  # an unrelated tmux shell or another user's process.
  if [[ "${pgid}" == "${pid}" ]]; then
    for attempt in $(seq 1 20); do
      kill -0 -- "-${pgid}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 -- "-${pgid}" 2>/dev/null; then
      echo "server process group ${pgid} still exists after KILL" >&2
      return 1
    fi
  fi
  wait "${pid}" 2>/dev/null || true
}

wait_server() {
  local pid="$1" port="$2" log="$3"
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server exited before listening on port ${port}; see ${log}" >&2
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" \
        >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "server timed out on port ${port}" >&2
  return 1
}

flush_server_cache() {
  local port="$1" benchmark="$2" method="$3" phase="$4"
  local response="" attempt
  for attempt in $(seq 1 3); do
    if response="$(curl --noproxy '*' -fsS -X POST \
        "http://127.0.0.1:${port}/flush_cache?timeout=60")" \
        && "${SGLANG_PYTHON}" -c \
          'import json,sys
body=sys.argv[1].strip()
try:
    ok=json.loads(body).get("success") is True
except (json.JSONDecodeError, AttributeError):
    ok=body.startswith("Cache flushed.")
raise SystemExit(0 if ok else 1)' \
          "${response}"; then
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${method}" "${phase}" "${attempt}" "${response}" \
        >>"${RUN_ROOT}/manifests/cache_flush_${benchmark}.tsv"
      return 0
    fi
    sleep 1
  done
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${method}" "${phase}" "failed" "${response}" \
    >>"${RUN_ROOT}/manifests/cache_flush_${benchmark}.tsv"
  echo "${benchmark}/${method}: cache flush did not return success=true (${phase})" >&2
  return 1
}

completed_arm_output() {
  local out="$1"
  [[ "${RESUME}" == 1 && -d "${out}" ]] || return 1
  "${TS_PYTHON}" -c '
import glob, json, sys
paths = glob.glob(sys.argv[1] + "/summary_*.json")
if len(paths) != 1:
    raise SystemExit(1)
data = json.load(open(paths[0], encoding="utf-8"))
req = data.get("request_log_summary") or {}
raise SystemExit(0 if data.get("n") == int(sys.argv[2]) and req.get("n_error") == 0 else 1)
' "${out}" "${EXPECTED_CASES}"
}

run_matrix() {
  local benchmark="$1" device="$2" server_port="$3" proxy_base="$4" bench_python="$5"
  local server_log="${RUN_ROOT}/logs/server_${benchmark}_card${device}_${server_port}.log"
  local server_pid
  start_server "${device}" "${server_port}" "${benchmark}"
  server_pid="${STARTED_SERVER_PID}"
  cleanup_server() { stop_server "${server_pid:-}"; }
  trap cleanup_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  wait_server "${server_pid}" "${server_port}" "${server_log}"
  printf 'timestamp\tmethod\tphase\tattempt\tresponse\n' \
    >"${RUN_ROOT}/manifests/cache_flush_${benchmark}.tsv"
  for index in "${!METHODS[@]}"; do
    local method="${METHODS[$index]}" arm="${ARMS[$index]}" recovery="${RECOVERY[$index]}"
    local out="${RUN_ROOT}/${PHASE}/${benchmark}/${method}"
    local proxy_port=$((proxy_base + index))
    if completed_arm_output "${out}"; then
      echo "SKIP completed ${benchmark}/${method}: ${EXPECTED_CASES} cases"
      continue
    fi
    if [[ -e "${out}" ]]; then
      out="${out}_retry_$(date +%Y%m%d_%H%M%S)"
      echo "preserving partial output; retrying ${benchmark}/${method} in ${out}"
    fi
    flush_server_cache "${server_port}" "${benchmark}" "${method}" before
    local common=(
      -m c2kv_eval.portable.run --benchmark "${benchmark}" --arm "${arm}"
      --upstream "http://127.0.0.1:${server_port}"
      --user-upstream "http://127.0.0.1:${server_port}"
      --proxy-port "${proxy_port}" --proxy-python "${SGLANG_PYTHON}"
      --out "${out}" --exact-out --run-name "${PHASE}_${benchmark}_${method}"
      --model "${MODEL}" --checkpoint "${CHECKPOINT}" --tokenizer "${TOKENIZER}"
      --reference-profile checkpoint-1088 --num-workers 1
    )
    if [[ -n "${recovery}" ]]; then common+=(--recovery-control "${recovery}"); fi
    echo "RUN ${benchmark}/${method}: arm=${arm} card=${device}"
    if [[ "${benchmark}" == tau2 ]]; then
      env PYTHONPATH="${TAU2_ROOT}/src:${BFCL_ROOT}" \
        "${bench_python}" "${common[@]}" --bench-python "${bench_python}" \
        --tau2-dir "${TAU2_ROOT}" --task-set airline --tau2-task-ids 3 \
        --tau2-num-trials 1 --tau2-max-steps 100 --tau2-timeout 1200
    else
      env PYTHONPATH="${BFCL_ROOT}:${TS_ROOT}" TS_PARALLEL="${TS_PARALLEL}" \
        TOOLSANDBOX_QWEN_TOOL_RESULT_COMPAT=1 \
        "${bench_python}" "${common[@]}" --bench-python "${bench_python}" \
        --toolsandbox-dir "${TS_ROOT}" \
        --ts-scenarios "${TS_SCENARIO}"
    fi
    # Make cross-arm isolation explicit and fail if the scheduler reports a
    # deferred/unsuccessful flush.  HTTP 200 alone is not sufficient because
    # the endpoint can return {"success": false} while requests are active.
    flush_server_cache "${server_port}" "${benchmark}" "${method}" after
  done
  stop_server "${server_pid}"
  server_pid=""
  trap - EXIT INT TERM
}

status=0
expected_completed=0
PID_TAU2=""
PID_TS=""
cleanup_workers() {
  local worker
  for worker in "${PID_TAU2}" "${PID_TS}"; do
    if [[ -n "${worker}" ]] && kill -0 "${worker}" 2>/dev/null; then
      kill -TERM "${worker}" 2>/dev/null || true
    fi
  done
  for worker in "${PID_TAU2}" "${PID_TS}"; do
    if [[ -n "${worker}" ]]; then wait "${worker}" 2>/dev/null || true; fi
  done
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${RUN_TAU2}" == 1 ]]; then
  run_matrix tau2 "${DEVICE_TAU2}" "${PORT_TAU2}" "${PROXY_BASE_TAU2}" "${TAU2_PYTHON}" &
  PID_TAU2=$!
  expected_completed=$((expected_completed + ${#METHODS[@]}))
fi
if [[ "${RUN_TOOLSANDBOX}" == 1 ]]; then
  run_matrix toolsandbox "${DEVICE_TS}" "${PORT_TS}" "${PROXY_BASE_TS}" "${TS_PYTHON}" &
  PID_TS=$!
  expected_completed=$((expected_completed + ${#METHODS[@]}))
fi
if [[ "${RUN_TAU2}" == 1 ]]; then
  wait "${PID_TAU2}" || status=1
  PID_TAU2=""
fi
if [[ "${RUN_TOOLSANDBOX}" == 1 ]]; then
  wait "${PID_TS}" || status=1
  PID_TS=""
fi
if [[ "${status}" -ne 0 ]]; then
  if [[ "${RUN_TAU2}" == 1 ]]; then
    echo "tau2 server log (tail):" >&2
    tail -n 160 "${RUN_ROOT}/logs/server_tau2_card${DEVICE_TAU2}_${PORT_TAU2}.log" >&2 || true
  fi
  if [[ "${RUN_TOOLSANDBOX}" == 1 ]]; then
    echo "ToolSandbox server log (tail):" >&2
    tail -n 160 "${RUN_ROOT}/logs/server_toolsandbox_card${DEVICE_TS}_${PORT_TS}.log" >&2 || true
  fi
  echo "at least one ${PHASE} matrix failed; inspect ${RUN_ROOT}" >&2
  exit "${status}"
fi
trap - EXIT INT TERM

env PYTHONPATH="${BFCL_ROOT}" "${TAU2_PYTHON}" \
  "${BFCL_ROOT}/c2kv_eval/portable/ops/summarize_compression_agnostic_recovery.py" \
  --root "${RUN_ROOT}" --out "${RUN_ROOT}/summary.csv" \
  --phase "${PHASE}" \
  --expected-completed "${expected_completed}" \
  --expected-cases "${EXPECTED_CASES}" --require-complete
touch "${RUN_ROOT}/${MARKER_PREFIX}_COMPLETE"
if env PYTHONPATH="${BFCL_ROOT}" "${TAU2_PYTHON}" \
    "${BFCL_ROOT}/c2kv_eval/portable/ops/summarize_compression_agnostic_recovery.py" \
    --root "${RUN_ROOT}" --out "${RUN_ROOT}/summary.csv" \
    --phase "${PHASE}" \
    --expected-completed "${expected_completed}" \
    --expected-cases "${EXPECTED_CASES}" --require-infrastructure-clean \
    --max-c2kv-pool-usage "${MAX_C2KV_POOL_USAGE}" \
    >"${RUN_ROOT}/infrastructure_gate.log" 2>&1; then
  touch "${RUN_ROOT}/${MARKER_PREFIX}_INFRA_CLEAN"
else
  echo "infrastructure quality gate failed; see ${RUN_ROOT}/infrastructure_gate.log" >&2
  exit 1
fi
if env PYTHONPATH="${BFCL_ROOT}" "${TAU2_PYTHON}" \
    "${BFCL_ROOT}/c2kv_eval/portable/ops/summarize_compression_agnostic_recovery.py" \
    --root "${RUN_ROOT}" --out "${RUN_ROOT}/summary.csv" \
    --phase "${PHASE}" \
    --expected-completed "${expected_completed}" \
    --expected-cases "${EXPECTED_CASES}" --require-complete --require-clean \
    >"${RUN_ROOT}/quality_gate.log" 2>&1; then
  touch "${RUN_ROOT}/${MARKER_PREFIX}_CLEAN"
  touch "${RUN_ROOT}/${MARKER_PREFIX}_TASK_CLEAN"
else
  echo "infrastructure clean; task-level non-normal terminations remain; see ${RUN_ROOT}/quality_gate.log" >&2
fi
echo "completed: ${RUN_ROOT}"
