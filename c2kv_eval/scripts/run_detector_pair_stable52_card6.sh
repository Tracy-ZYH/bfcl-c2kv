#!/usr/bin/env bash
# Sequential detector-only comparison, same stable52 split and Replace W2.
set -Eeuo pipefail
ROOT=/home/zhuyuhan/project/bfcl-c2kv
PAIR_ROOT="${PAIR_ROOT:-/home/zhuyuhan/zh_runs/detector_pair_stable52_card6_$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-rule,trained}"
[[ ! -e "$PAIR_ROOT" ]] || { echo "Refusing existing output: $PAIR_ROOT" >&2; exit 2; }
IFS=, read -ra variants <<< "$VARIANTS"
for variant in "${variants[@]}"; do
  [[ "$variant" == rule || "$variant" == trained ]] || exit 2
done
mkdir -p "$PAIR_ROOT"
for variant in "${variants[@]}"; do
  env ROOT="$ROOT" DETECTOR_VERSION=3 DETECTOR_VARIANT="$variant" \
    RUN_ROOT="$PAIR_ROOT/$variant" CLEAN_OUTPUT=0 FAST_DEV=0 FOLDS=5 MAX_EXAMPLES=52 \
    DEVICES=6 PORTS=35766 RUN_HELDOUT_CONTROLS=1 \
    CLOSED_LOOP_RATES=0.45,0.65,0.85 ONLINE_SHADOW=1 \
    REQUEST_CANDIDATE_LOGPROBS=0 REPAIR_ARM=d_corr_replace_w2 REPAIR_WINDOW=2 \
    BFCL_PYTHON=/home/liuyancheng/envs/sgl/bin/python \
    SGLANG_PYTHON=/home/liuyancheng/envs/sgl/bin/python \
    SGLANG_ROOT=/home/zhuyuhan/project/kvoffload-sglang-c2kv \
    FEATURES_CSV=/home/zhuyuhan/zh_restore/runs/detector_v2_regen/detector_cv/feature_benchmark/detector_features.csv \
    MEM_FRACTION_STATIC=0.45 C2KV_POOL_FRACTION=0.10 \
    SGLANG_EXTRA_ARGS=--disable-cuda-graph \
    PYTHONPATH="$ROOT:/home/zhuyuhan/project/kvoffload-sglang-c2kv/python:/home/zhuyuhan/project/c2kv:/home/zhuyuhan/project/python_overlay" \
    bash "$ROOT/c2kv_eval/scripts/run_bfcl_combined_logistic_v2_replace_w2.sh" \
    2>&1 | tee "$PAIR_ROOT/${variant}.log"
done
env PYTHONPATH="$ROOT" /home/liuyancheng/envs/sgl/bin/python \
  -m c2kv_eval.analysis.summarize_detector_pair --root "$PAIR_ROOT"
echo "Completed: $PAIR_ROOT"
