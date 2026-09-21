#!/usr/bin/env bash
set -euo pipefail
# Ignore any inherited logical-device remapping. GPU values below are physical IDs.
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Keep server readiness checks and benchmark traffic on loopback even when the
# login shell exports an HTTP(S) proxy.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
BFCL_DIR="${BFCL_DIR:-${ROOT}}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/outputs/qwen3_4b_agent_c2kv_60k_248/sglang_ckpt1876}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/crossbackend_racer_$(date +%Y%m%d_%H%M%S)}"
CASES="${CASES:-2}"
CATEGORY="${CATEGORY:-multi_turn_base}"
IDS_PATH="${IDS_PATH:-${ROOT}/results/multi_turn_base_full200/ground_truth/episode_ids.txt}"
[[ -s "${IDS_PATH}" ]] || { echo "Missing fixed BFCL IDs: ${IDS_PATH}" >&2; exit 2; }
RUN_IDS="$(head -n "${CASES}" "${IDS_PATH}" | paste -sd, -)"
[[ -n "${RUN_IDS}" ]] || { echo "No BFCL IDs selected for CASES=${CASES}" >&2; exit 2; }
# Use a run-specific proxy range so stale or concurrent portable proxies cannot collide.
PROXY_PORT_BASE="${PROXY_PORT_BASE:-$((40000 + ($$ % 1000) * 10))}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${HOME}/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
CUDA_HOME_FOR_JIT="${CUDA_HOME_FOR_JIT:-/usr/local/cuda-13.0}"
CC_BIN="${CC_BIN:-/usr/bin/gcc-12}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-12}"
NVCC_CCBIN="${NVCC_CCBIN:-${CXX_BIN}}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
[[ -d "${BFCL_DIR}/bfcl_eval/data" ]] || { echo "Missing BFCL official data: ${BFCL_DIR}/bfcl_eval/data" >&2; exit 2; }
BFCL_PYTHON="${BFCL_PYTHON:-${HOME}/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
[[ ! -e "${RESULT_ROOT}" ]] || { echo "Refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
mkdir -p "${RESULT_ROOT}/servers" "${RESULT_ROOT}/runs" "${RESULT_ROOT}/summary"
PIDS=()
cleanup() {
  set +e
  # Signal only the server parent first so it can drain and reap its workers.
  for p in "${PIDS[@]:-}"; do kill -TERM "$p" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    alive=0
    for p in "${PIDS[@]:-}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break
    sleep 0.5
  done
  for p in "${PIDS[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true; fi
  done
  for p in "${PIDS[@]:-}"; do wait "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
GPU=(0 1 2 3); PORT=(34740 34750 34760 34770)
BACKEND=(c2kv h2o snapkv pyramidkv)
# GPU0 runs C2KV and StreamingLLM sequentially; other GPUs run one backend.
ARMS=("c2kv c2kv_racer streamingllm_r25 streamingllm_r25_racer" "h2o_r25 h2o_r25_racer" "snapkv_r25 snapkv_r25_racer" "pyramidkv_r25 pyramidkv_r25_racer")
[[ -x "${CC_BIN}" && -x "${CXX_BIN}" ]] || { echo "gcc-12/g++-12 required; set CC_BIN/CXX_BIN" >&2; exit 1; }
for i in "${!GPU[@]}"; do
  echo "starting ${BACKEND[$i]} on physical GPU ${GPU[$i]} (serialized JIT startup)"
  setsid env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU[$i]}" \
    CUDA_HOME="${CUDA_HOME_FOR_JIT}" CUDACXX="${CUDA_HOME_FOR_JIT}/bin/nvcc" \
    PATH="${CUDA_HOME_FOR_JIT}/bin:${PATH}" \
    CC="${CC_BIN}" CXX="${CXX_BIN}" NVCC_CCBIN="${NVCC_CCBIN}" CUDAHOSTCXX="${CXX_BIN}" \
    MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
    PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}" "${SGLANG_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${PORT[$i]}" \
    --skip-server-warmup \
    --tool-call-parser qwen25 \
    --enable-return-hidden-states \
    --enable-c2kv --enable-streaming-session --c2kv-gist-type dynamic-interleave \
    --c2kv-gist-param qkv --c2kv-query-proj base \
    --attention-backend "${ATTENTION_BACKEND}" \
    --decode-attention-backend "${ATTENTION_BACKEND}" \
    --prefill-attention-backend "${ATTENTION_BACKEND}" \
    --sampling-backend "${SAMPLING_BACKEND}" \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    >"${RESULT_ROOT}/servers/${BACKEND[$i]}.log" 2>&1 & PIDS+=("$!")
  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl --noproxy '*' -fsS "http://127.0.0.1:${PORT[$i]}/health" >/dev/null 2>&1; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "server ${BACKEND[$i]} exited during startup; see ${RESULT_ROOT}/servers/${BACKEND[$i]}.log" >&2
      exit 1
    fi
    (( SECONDS < deadline )) || { echo "server ${BACKEND[$i]} not ready; see ${RESULT_ROOT}/servers/${BACKEND[$i]}.log" >&2; exit 1; }
    sleep 2
  done
done
export PYTHONPATH="${ROOT}:${SGLANG_ROOT}/python:${PYTHONPATH:-}"
run_gpu_arms() {
  local i="$1" local_index=0 arm out proxy_port
  local arms_for_gpu=()
  read -ra arms_for_gpu <<<"${ARMS[$i]}"
  for arm in "${arms_for_gpu[@]}"; do
    out="${RESULT_ROOT}/runs/${arm}"
    proxy_port=$((PROXY_PORT_BASE + i * 4 + local_index))
    local_index=$((local_index + 1))
    echo "[gpu ${GPU[$i]}] running ${arm} via proxy port ${proxy_port}"
    "${BFCL_PYTHON}" -m c2kv_eval.portable.run --benchmark bfcl --categories "${CATEGORY}" --arm "$arm" --upstream "http://127.0.0.1:${PORT[$i]}" --model c2kv-agent --proxy-port "${proxy_port}" --checkpoint "${MODEL_PATH}" --tokenizer "${TOKENIZER_PATH}" --bfcl-dir "${BFCL_DIR}" --run-ids "${RUN_IDS}" --num-workers 1 --max-tasks "${CASES}" --out "$out" --exact-out 2>&1 | tee "${RESULT_ROOT}/runs/${arm}.launcher.log"
  done
}
WORKER_PIDS=()
for i in "${!PORT[@]}"; do
  run_gpu_arms "$i" & WORKER_PIDS+=("$!")
done
worker_status=0
for p in "${WORKER_PIDS[@]}"; do
  wait "$p" || worker_status=1
done
(( worker_status == 0 )) || { echo "one or more GPU arm workers failed" >&2; exit 1; }
"${BFCL_PYTHON}" - "${RESULT_ROOT}" <<'PYEND'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for p in sorted((root/'runs').glob('**/summary_*.json')):
 x=json.loads(p.read_text()); rl=x.get('request_log_summary') or {}; rs=[]
 q=Path(x.get('request_log',''))
 if q.exists():
  for line in q.read_text().splitlines():
   try: r=json.loads(line)
   except Exception: continue
   if isinstance(r,dict) and r.get('status')=='ok': rs.append(r)
 arm=x.get('arm',''); racer=arm.endswith('_racer')
 rows.append({'arm':arm,'category':x.get('categories') or x.get('category'),'official_score':x.get('official_score'),'official_correct_count':x.get('official_correct_count'),'official_total_count':x.get('official_total_count'),'racer_enabled':racer,'detector_calls':sum(int(r.get('detector_calls') or 0) for r in rs),'detector_trigger_count':sum(bool(r.get('triggered')) for r in rs),'recovery_count':sum(int(r.get('recovery_count') or 0) for r in rs),'restored_raw_tokens':sum(int(r.get('restored_raw_tokens') or 0) for r in rs),'request_errors':rl.get('n_error'),'summary_path':str(p)})
out=root/'summary'/'crossbackend_racer.csv'
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['arm']); w.writeheader(); w.writerows(rows)
(root/'summary'/'crossbackend_racer.json').write_text(json.dumps(rows,indent=2))
failures=[f"{r['arm']}: request_errors={r['request_errors']}" for r in rows if int(r.get('request_errors') or 0)>0]
failures += [f"{r['arm']}: detector_calls=0" for r in rows if r.get('racer_enabled') and int(r.get('detector_calls') or 0)==0]
if failures:
 print('cross-backend validation failed: ' + '; '.join(failures), file=sys.stderr)
 raise SystemExit(1)
print(f'wrote {len(rows)} rows to {out}')
PYEND
echo "completed; results: ${RESULT_ROOT}"
