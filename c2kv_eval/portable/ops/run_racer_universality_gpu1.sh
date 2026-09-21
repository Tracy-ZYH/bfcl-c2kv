#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env GPU_ID=1 SERVER_PORT=34750 ARMS_SPEC="h2o_r25 h2o_r25_racer" HISTORY_KV_TARGET_TOKENS="${HISTORY_KV_TARGET_TOKENS:-768}" MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}" MAX_DOC_NUM="${MAX_DOC_NUM:-16}" bash "${SCRIPT_DIR}/run_racer_universality_one_gpu.sh" "$@"
