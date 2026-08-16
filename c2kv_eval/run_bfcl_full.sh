#!/usr/bin/env bash
set -euo pipefail

BFCL_ROOT=/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard
RUN_ROOT=/home/zhuyuhan/project/gorilla/bfcl_runs/qwen3_4b_full
TOKENIZER=/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-32000}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507-FC}"
CATEGORY="${CATEGORY:-simple_python}"
THREADS="${THREADS:-1}"

mkdir -p "$RUN_ROOT/logs"

LOG="$RUN_ROOT/logs/${CATEGORY}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

cd "$BFCL_ROOT"

echo "===== BFCL FULL ====="
echo "model=$MODEL"
echo "category=$CATEGORY"
echo "server=http://${HOST}:${PORT}/v1"
echo "log=$LOG"

echo
echo "[1] Check server"
curl --noproxy '*' \
  --connect-timeout 5 \
  -fsS \
  "http://${HOST}:${PORT}/v1/models"

echo
echo "[2] Configure BFCL"

export BFCL_PROJECT_ROOT="$RUN_ROOT"

export LOCAL_SERVER_ENDPOINT="$HOST"
export LOCAL_SERVER_PORT="$PORT"

export REMOTE_OPENAI_BASE_URL="http://${HOST}:${PORT}/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export REMOTE_OPENAI_TOKENIZER_PATH="$TOKENIZER"

export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

echo
echo "[3] Generate"

bfcl generate \
  --model "$MODEL" \
  --test-category "$CATEGORY" \
  --skip-server-setup \
  --num-threads "$THREADS"

echo
echo "[4] Evaluate"

bfcl evaluate \
  --model "$MODEL" \
  --test-category "$CATEGORY"

echo
echo "===== DONE ====="
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG=$LOG"
