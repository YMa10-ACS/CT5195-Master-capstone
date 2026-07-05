#!/usr/bin/env bash
set -e

CLOUD_TARGET="${1:-${CLOUD_TARGET:-127.0.0.1}}"

for mode in http https-aes-gcm https-chacha20; do
  echo "===== Running mode: $mode ====="
  bash run_edge_comparison.sh "$mode" "$CLOUD_TARGET"
done
