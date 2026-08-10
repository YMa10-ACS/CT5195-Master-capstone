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


if [ "$HTTP_MODE" = "http" ]; then
  PORT=8000
elif [ "$HTTP_MODE" = "https-aes-gcm" ]; then
  PORT=8443
  OPENSSL_CONFIG="$ROOT/openssl_tls13_aes.cnf"
elif [ "$HTTP_MODE" = "https-chacha20" ]; then
  PORT=8444
  OPENSSL_CONFIG="$ROOT/openssl_tls13_chacha20.cnf"
else
  echo "Unsupported cloud mode: $HTTP_MODE" >&2
  exit 1
fi

mkdir -p "$ROOT/record"

CLOUD_LOG="$ROOT/record/distributed_vlm_cloud_${HTTP_MODE}.log"
CLOUD_PID="$ROOT/record/distributed_vlm_cloud_${HTTP_MODE}.pid"

cd "$ROOT"

if [ "$HTTP_MODE" == "http" ]; then
  gunicorn -k gthread -w 1 --threads 1 \
    --keep-alive 30 \
    -b ${CLOUD_HOST}:${PORT} \
    "cloud:create_app()" \
    2>&1 | tee "$CLOUD_LOG" &
else
  env OPENSSL_CONF="$OPENSSL_CONFIG" \
    gunicorn -k gthread -w 1 --threads 1 \
    --keep-alive 30 \
    -b "${CLOUD_HOST}:${PORT}" \
    --certfile ../cert/cert.pem \
    --keyfile ../cert/key.pem \
    "cloud:create_app()" \
    2>&1 | tee "$CLOUD_LOG" &
fi

echo "$!" > "$CLOUD_PID"
echo "cloud mode: $HTTP_MODE"
echo "cloud host: $CLOUD_HOST"
echo "cloud log: $CLOUD_LOG"
echo "cloud pid: $(cat "$CLOUD_PID")"
