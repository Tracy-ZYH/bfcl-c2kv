#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env GPU_ID=3 SERVER_PORT=34770 ARMS_SPEC="pyramidkv_r25 pyramidkv_r25_racer" HISTORY_KV_TARGET_TOKENS="${HISTORY_KV_TARGET_TOKENS:-768}" MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}" MAX_DOC_NUM="${MAX_DOC_NUM:-16}" bash "${SCRIPT_DIR}/run_racer_universality_one_gpu.sh" "$@"
