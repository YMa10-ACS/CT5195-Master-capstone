#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-lusxvr/nanoVLM-230M-8k}"
PROMPT="${PROMPT:-What is this?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
HTTP_MODE="${1:-${HTTP_MODE:-https-aes-gcm}}"
CLOUD_HOST="${CLOUD_HOST:-0.0.0.0}"

mkdir -p "$ROOT/record"

CLOUD_LOG="$ROOT/record/distributed_vlm_cloud_${HTTP_MODE}.log"
CLOUD_PID="$ROOT/record/distributed_vlm_cloud_${HTTP_MODE}.pid"

cd "$ROOT"

python3 ./cloud.py \
  --transport "$HTTP_MODE" \
  --host "$CLOUD_HOST" \
  --weights "$MODEL" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$CLOUD_LOG" &

echo "$!" > "$CLOUD_PID"
echo "cloud mode: $HTTP_MODE"
echo "cloud host: $CLOUD_HOST"
echo "cloud log: $CLOUD_LOG"
echo "cloud pid: $(cat "$CLOUD_PID")"
