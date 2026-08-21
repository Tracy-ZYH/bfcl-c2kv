#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
RATIO="${RATIO:-4}"
DEVICE="${DEVICE:-7}"
PORT="${PORT:-33607}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_$(date +%Y%m%d_%H%M%S)}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

MODE="history_full_closed_loop"
REFERENCE_ROOT="${RUN_ROOT}/reference_rollout"
VERIFY_ROOT="${RUN_ROOT}/stability_rerun"
FROZEN_ROOT="${RUN_ROOT}/frozen_reference"
SERVER_PID=""

log_info() {
  echo "[$(date '+%F %T')] $*"
}

source_env_file() {
  local path="$1"
  local rc
  log_info "source ${path}"
  set +e +u
  source "${path}"
  rc=$?
  set -e
  set +u
  if [ "${rc}" -ne 0 ]; then
    log_info "source failed: ${path} rc=${rc}"
    return "${rc}"
  fi
}

cleanup() {
  local status=$?
  set +e
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

check_port_free() {
  local port="$1"
  set +e
  "${BFCL_PYTHON}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
    finally:
        sock.close()
except PermissionError as exc:
    print(f"[WARN] socket permission check failed for port {port}: {exc}", file=sys.stderr)
    sys.exit(0)
sys.exit(0 if result != 0 else 1)
PY
  local rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    echo "Port ${port} is already in use. Stop the existing server or set PORT."
    return 1
  fi
}

prepare_dirs() {
  if [ "${CLEAN_OUTPUT}" = "1" ]; then
    rm -rf "${REFERENCE_ROOT:?}" "${VERIFY_ROOT:?}" "${FROZEN_ROOT:?}"
  fi
  mkdir -p \
    "${REFERENCE_ROOT}/${MODE}/result" \
    "${REFERENCE_ROOT}/${MODE}/score" \
    "${REFERENCE_ROOT}/${MODE}/logs" \
    "${VERIFY_ROOT}/${MODE}/result" \
    "${VERIFY_ROOT}/${MODE}/score" \
    "${VERIFY_ROOT}/${MODE}/logs" \
    "${FROZEN_ROOT}"
}

start_server() {
  local log="${RUN_ROOT}/server_${DEVICE}_${PORT}.log"
  (
    cd "${SGLANG_ROOT}"
    SGLANG_DEBUG_MEMORY_POOL=1 \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_EMPTY_CACHE_INTERVAL=1 \
    ASCEND_LAUNCH_BLOCKING=1 \
    TASK_QUEUE_ENABLE=1 \
    no_proxy='*' \
    NO_PROXY='*' \
    http_proxy='' \
    https_proxy='' \
    HTTP_PROXY='' \
    HTTPS_PROXY='' \
    ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
    exec "${SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${MODEL_ID}" \
      --model-impl sglang \
      --device npu \
      --attention-backend ascend \
      --tool-call-parser qwen25 \
      --enable-c2kv \
      --dtype bfloat16 \
      --mem-fraction-static 0.55 \
      --host 127.0.0.1 \
      --port "${PORT}"
  ) > "${log}" 2>&1 &
  SERVER_PID="$!"
  log_info "[server] device=${DEVICE} port=${PORT} pid=${SERVER_PID} log=${log}"
}

wait_health() {
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      log_info "[server] crashed; last log:"
      tail -n 120 "${RUN_ROOT}/server_${DEVICE}_${PORT}.log" || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log_info "[server] healthy on port ${PORT}"
      return 0
    fi
    if [ $((attempt % 15)) -eq 0 ]; then
      log_info "[server] waiting for health on port ${PORT} (${attempt}/900)"
    fi
    sleep 2
  done
  log_info "[server] health check timed out"
  tail -n 120 "${RUN_ROOT}/server_${DEVICE}_${PORT}.log" || true
  return 1
}

run_full_rollout() {
  local root="$1"
  local ids_path="$2"
  local reference_details="$3"
  local log="${root}/${MODE}/logs/run.log"
  local cmd=(
    "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_drift
    --mode "${MODE}"
    --category "${CATEGORY}"
    --max-examples "${MAX_EXAMPLES}"
    --model "${MODEL_ID}"
    --served-model-name "${MODEL_ID}"
    --base-url "http://127.0.0.1:${PORT}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --result-dir "${root}/${MODE}/result"
    --details-path "${root}/${MODE}/logs/details.jsonl"
    --metrics-path "${root}/${MODE}/logs/drift_metrics.jsonl"
    --summary-path "${root}/${MODE}/logs/run_summary.json"
    --ratio "${RATIO}"
    --temperature 0
  )
  if [ -n "${ids_path}" ]; then
    cmd+=(--ids-path "${ids_path}")
  fi
  if [ -n "${reference_details}" ]; then
    cmd+=(--reference-details-path "${reference_details}")
  fi
  (
    cd "${ROOT}"
    exec "${cmd[@]}"
  ) > "${log}" 2>&1
  log_info "[runner] done root=${root} log=${log}"
}

evaluate_rollout() {
  local root="$1"
  local partial="$2"
  local cmd=(
    "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner
    --model "${MODEL_ID}"
    --test-category "${CATEGORY}"
    --result-dir "${root}/${MODE}/result"
    --score-dir "${root}/${MODE}/score"
  )
  if [ "${partial}" = "1" ]; then
    cmd+=(--partial-eval)
  fi
  (
    cd "${ROOT}"
    exec "${cmd[@]}"
  ) > "${root}/${MODE}/logs/eval.log" 2>&1
  log_info "[eval] done root=${root}"
}

export_success_ids() {
  local root="$1"
  local output="$2"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" c2kv_eval/analysis/export_success_ids.py \
      --run-root "${root}" \
      --mode "${MODE}" \
      --category "${CATEGORY}" \
      --output-path "${output}"
  ) > "${root}/${MODE}/logs/export_success_ids.log" 2>&1
  log_info "[ids] wrote ${output}"
}

write_summary() {
  "${BFCL_PYTHON}" - "${RUN_ROOT}" "${CATEGORY}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
category = sys.argv[2]
mode = "history_full_closed_loop"

def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def first(root, pattern):
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None

def score(root):
    path = first(root / mode / "score", f"*_{category}_score.json")
    rows = load_jsonl(path) if path else []
    return rows[0] if rows else {}

def count_lines(path):
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

reference_root = run_root / "reference_rollout"
verify_root = run_root / "stability_rerun"
frozen = run_root / "frozen_reference"
frozen_ids = frozen / "correct_ids.txt"
verify_ids = run_root / "stability_verified_ids.txt"

summary = {
    "run_root": str(run_root),
    "category": category,
    "temperature": 0,
    "reference_score": score(reference_root),
    "stability_score": score(verify_root),
    "frozen_success_count": count_lines(frozen_ids),
    "stability_success_count": count_lines(verify_ids),
    "stable_100_percent": (
        count_lines(frozen_ids) > 0
        and count_lines(frozen_ids) == count_lines(verify_ids)
    ),
    "frozen_reference_details": str(frozen / "details.jsonl"),
    "frozen_reference_ids": str(frozen_ids),
}
(run_root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
lines = [
    "# BFCL Full Temperature-0 Stability",
    "",
    f"Run root: `{run_root}`",
    "",
    "| Pass | Accuracy | Correct | Total |",
    "| --- | ---: | ---: | ---: |",
]
for name, row in [
    ("Reference rollout", summary["reference_score"]),
    ("Stability rerun on frozen ids", summary["stability_score"]),
]:
    lines.append(
        "| {name} | {acc} | {correct} | {total} |".format(
            name=name,
            acc=row.get("accuracy", "-"),
            correct=row.get("correct_count", "-"),
            total=row.get("total_count", "-"),
        )
    )
lines.extend(
    [
        "",
        f"Frozen success ids: `{frozen_ids}`",
        f"Frozen Full details: `{frozen / 'details.jsonl'}`",
        f"Stable 100 percent: `{summary['stable_100_percent']}`",
    ]
)
(run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

log_info "BFCL Full temperature=0 stability run starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES}"
log_info "DEVICE=${DEVICE} PORT=${PORT}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh
prepare_dirs
check_port_free "${PORT}"
start_server
wait_health

run_full_rollout "${REFERENCE_ROOT}" "" ""
evaluate_rollout "${REFERENCE_ROOT}" "0"
export_success_ids "${REFERENCE_ROOT}" "${FROZEN_ROOT}/correct_ids.txt"
cp "${REFERENCE_ROOT}/${MODE}/logs/details.jsonl" "${FROZEN_ROOT}/details.jsonl"
cp "${REFERENCE_ROOT}/${MODE}/logs/run_summary.json" "${FROZEN_ROOT}/run_summary.json"
cp "${REFERENCE_ROOT}/${MODE}/logs/export_success_ids.log" "${FROZEN_ROOT}/export_success_ids.log"

run_full_rollout \
  "${VERIFY_ROOT}" \
  "${FROZEN_ROOT}/correct_ids.txt" \
  "${FROZEN_ROOT}/details.jsonl"
evaluate_rollout "${VERIFY_ROOT}" "1"
export_success_ids "${VERIFY_ROOT}" "${RUN_ROOT}/stability_verified_ids.txt"
write_summary

log_info "BFCL Full temperature=0 stability run complete"
log_info "Report: ${RUN_ROOT}/report.md"
log_info "Frozen reference: ${FROZEN_ROOT}"
