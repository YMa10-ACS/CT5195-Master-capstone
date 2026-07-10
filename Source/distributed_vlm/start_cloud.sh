#!/usr/bin/env bash
###
 # @Description: 
 # @Date: 2026-07-06 00:28:23
 # @Author: Yaoquan Ma
### 

ROOT="$(cd "$(dirname "$0")" && pwd)"
export WEIGHTS="${WEIGHTS:-lusxvr/nanoVLM-230M-8k}"
export PROMPT="${PROMPT:-What is this?}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
CLOUD_HOST="${CLOUD_HOST:-0.0.0.0}"
HTTP_MODE="${1:-${HTTP_MODE:-https-aes-gcm}}"


if [ "$HTTP_MODE" == "http" ]; then
	PORT=8000
elif [ "$HTTP_MODE" == "https-chacha20" ]; then
	PORT=8444
	CIPHER="ECDHE-RSA-CHACHA20-POLY1305"
else 
	PORT=8443
	CIPHER="ECDHE-RSA-AES128-GCM-SHA256"
fi

mkdir -p "$ROOT/record"

CLOUD_LOG="$ROOT/record/distributed_vlm_cloud_${HTTP_MODE}.log"
CLOUD_PID="$ROOT/record/distributed_vlm_cloud_${HTTP_MODE}.pid"

cd "$ROOT"

if [ "$HTTP_MODE" == "http" ]; then
  gunicorn -k gthread -w 1 --threads 4 \
    --keep-alive 30 \
    -b ${CLOUD_HOST}:${PORT} \
    "cloud:create_app()" \
    2>&1 | tee "$CLOUD_LOG" &
else
  gunicorn -k gthread -w 1 --threads 4 \
    --keep-alive 30 \
    -b ${CLOUD_HOST}:${PORT} \
    --certfile ../cert/cert.pem \
    --keyfile ../cert/key.pem \
    --ciphers ${CIPHER} \
    "cloud:create_app()" \
    2>&1 | tee "$CLOUD_LOG" &
fi

echo "$!" > "$CLOUD_PID"
echo "cloud mode: $HTTP_MODE"
echo "cloud host: $CLOUD_HOST"
echo "cloud log: $CLOUD_LOG"
echo "cloud pid: $(cat "$CLOUD_PID")"
