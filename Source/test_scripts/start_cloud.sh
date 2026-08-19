#!/usr/bin/env bash
###
 # @Description: 
 # @Date: 2026-07-06 00:28:23
 # @Author: Yaoquan Ma
### 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$SOURCE_DIR/distributed_vlm"
CONFIG_DIR="$SOURCE_DIR/configs"
SECRETS_DIR="$SOURCE_DIR/secrets"
RECORD_DIR="$SOURCE_DIR/record"
GUNICORN="$SOURCE_DIR/.venv/bin/gunicorn"
export TDX_ATTEST_HELPER="${TDX_ATTEST_HELPER:-$SOURCE_DIR/tdx_attest/tdx-attest}"
export WEIGHTS="${WEIGHTS:-lusxvr/nanoVLM-230M-8k}"
export PROMPT="${PROMPT:-What is this?}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
CLOUD_HOST="${CLOUD_HOST:-0.0.0.0}"
HTTP_MODE="${1:-${HTTP_MODE:-https-aes-gcm}}"


if [ "$HTTP_MODE" = "http" ]; then
  PORT=8000
elif [ "$HTTP_MODE" = "https-aes-gcm" ]; then
  PORT=8443
  OPENSSL_CONFIG="$CONFIG_DIR/openssl_tls13_aes.cnf"
elif [ "$HTTP_MODE" = "https-chacha20" ]; then
  PORT=8444
  OPENSSL_CONFIG="$CONFIG_DIR/openssl_tls13_chacha20.cnf"
else
  echo "Unsupported cloud mode: $HTTP_MODE" >&2
  exit 1
fi

mkdir -p "$RECORD_DIR"

CLOUD_LOG="$RECORD_DIR/distributed_vlm_cloud_${HTTP_MODE}.log"
CLOUD_PID="$RECORD_DIR/distributed_vlm_cloud_${HTTP_MODE}.pid"

cd "$APP_DIR"

if [ "$HTTP_MODE" == "http" ]; then
  "$GUNICORN" -k gthread -w 1 --threads 1 \
    --keep-alive 30 \
    -b ${CLOUD_HOST}:${PORT} \
    "cloud:create_app()" \
    2>&1 | tee "$CLOUD_LOG" &
else
  env OPENSSL_CONF="$OPENSSL_CONFIG" \
    "$GUNICORN" -k gthread -w 1 --threads 1 \
    --keep-alive 30 \
    -b "${CLOUD_HOST}:${PORT}" \
    --certfile "$SECRETS_DIR/cert.pem" \
    --keyfile "$SECRETS_DIR/key.pem" \
    "cloud:create_app()" \
    2>&1 | tee "$CLOUD_LOG" &
fi

echo "$!" > "$CLOUD_PID"
echo "cloud mode: $HTTP_MODE"
echo "cloud host: $CLOUD_HOST"
echo "cloud log: $CLOUD_LOG"
echo "cloud pid: $(cat "$CLOUD_PID")"
