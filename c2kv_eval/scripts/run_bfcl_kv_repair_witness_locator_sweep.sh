#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-52}"
RATIO="${RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
DEVICES="${DEVICES:-5,6}"
PORTS="${PORTS:-35250,35260}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_kv_repair_witness_locator_$(date +%Y%m%d_%H%M%S)}"

ARMS="${ARMS:-d_corr_replace_w1,d_corr_replace_w1_first,d_corr_replace_w1_witness,d_corr_replace_w2,d_corr_replace_w4}"
REPAIR_TRIGGER="${REPAIR_TRIGGER:-oracle}"
REPAIR_LOCATOR="${REPAIR_LOCATOR:-recent}"
WITNESS_CORE_PATH="${WITNESS_CORE_PATH:-/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_witness_core.py}"

CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
RATIO="${RATIO}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
ARMS="${ARMS}" \
REPAIR_TRIGGER="${REPAIR_TRIGGER}" \
REPAIR_LOCATOR="${REPAIR_LOCATOR}" \
WITNESS_CORE_PATH="${WITNESS_CORE_PATH}" \
RUN_ROOT="${RUN_ROOT}" \
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}" \
RUN_COMPARE=1 \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh"

echo "Witness locator summary: ${RUN_ROOT}/kv_repair_witness_summary.csv"
