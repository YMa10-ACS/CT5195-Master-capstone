#!/usr/bin/env bash
###
 # @Description: 
 # @Date: 2026-07-06 19:18:57
 # @Author: Yaoquan Ma
### 
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
TDX_ATTESTATION_PATH="${ROOT}/tdx_attest"
CLOUD_NO_TDX_TARGET="10.128.0.7"
CLOUD_TDX_TARGET="10.128.0.6"
MODES="http https-aes-gcm https-chacha20"

clear_current_records() {
  mkdir -p "$ROOT/record"
  for mode in $MODES; do
    rm -f "$ROOT/record/distributed_vlm_edge_${mode}.log"
    rm -f "$ROOT/record/distributed_vlm_metrics_${mode}.csv"
    rm -f "$ROOT/record/distributed_vlm_throughput_${mode}.csv"
  done
}

rename_records_for_target() {
  target="$1"
  label="${target//./_}"

  for mode in $MODES; do
    edge_log="$ROOT/record/distributed_vlm_edge_${mode}.log"
    metrics_csv="$ROOT/record/distributed_vlm_metrics_${mode}.csv"
    throughput_csv="$ROOT/record/distributed_vlm_throughput_${mode}.csv"

    if [ -f "$edge_log" ]; then
      mv "$edge_log" "$ROOT/record/distributed_vlm_edge_${mode}_${label}.log"
    fi

    if [ -f "$metrics_csv" ]; then
      mv "$metrics_csv" "$ROOT/record/distributed_vlm_metrics_${mode}_${label}.csv"
    fi

    if [ -f "$throughput_csv" ]; then
      mv "$throughput_csv" "$ROOT/record/distributed_vlm_throughput_${mode}_${label}.csv"
    fi
  done
}

run_split_inference() {
  target="$1"
  attestation="$2"
  clear_current_records

  for mode in $MODES; do
    echo "===== Running mode: $mode split target: $target attestation: $attestation ====="
    ATTESTATION="$attestation" bash run_edge_comparison.sh "$mode" "$target" "split"
  done

  rename_records_for_target "$target"
}

run_local_inference() {
  clear_current_records
  echo "===== Running mode: $mode local ====="
  bash run_edge_comparison.sh "http" "127.0.0.1" "local"
  rename_records_for_target "local"
}

build_tdx_attestation() {
  (cd "$TDX_ATTESTATION_PATH" && ./build.sh)
}


cd "$ROOT"

build_tdx_attestation

run_split_inference "$CLOUD_NO_TDX_TARGET" disabled
run_split_inference "$CLOUD_TDX_TARGET" enabled

run_local_inference
