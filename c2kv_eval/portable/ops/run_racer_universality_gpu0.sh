#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env GPU_ID=0 SERVER_PORT=34740 ARMS_SPEC="c2kv c2kv_racer streamingllm_r25 streamingllm_r25_racer" HISTORY_KV_TARGET_TOKENS="${HISTORY_KV_TARGET_TOKENS:-768}" MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}" MAX_DOC_NUM="${MAX_DOC_NUM:-16}" bash "${SCRIPT_DIR}/run_racer_universality_one_gpu.sh" "$@"
