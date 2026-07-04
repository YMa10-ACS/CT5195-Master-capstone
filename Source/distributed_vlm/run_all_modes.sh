#!/usr/bin/env bash
set -e

for mode in http https-aes-gcm https-chacha20; do
  echo "===== Running mode: $mode ====="
  bash comparison_new.sh "$mode"
done
