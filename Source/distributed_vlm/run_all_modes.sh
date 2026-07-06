#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CLOUD_NO_TDX_TARGET="10.128.0.7"
CLOUD_TDX_TARGET="10.128.0.6"
MODES="http https-aes-gcm https-chacha20"

clear_current_records() {
  mkdir -p "$ROOT/record"
  for mode in $MODES; do
    rm -f "$ROOT/record/distributed_vlm_edge_${mode}.log"
    rm -f "$ROOT/record/distributed_vlm_metrics_${mode}.csv"
  done
}

rename_records_for_target() {
  target="$1"
  label="${label//./_}"

  for mode in $MODES; do
    edge_log="$ROOT/record/distributed_vlm_edge_${mode}.log"
    metrics_csv="$ROOT/record/distributed_vlm_metrics_${mode}.csv"

    if [ -f "$edge_log" ]; then
      mv "$edge_log" "$ROOT/record/distributed_vlm_edge_${mode}_${label}.log"
    fi

    if [ -f "$metrics_csv" ]; then
      mv "$metrics_csv" "$ROOT/record/distributed_vlm_metrics_${mode}_${label}.csv"
    fi
  done
}

run_split_inference() {
  target="$1"
  clear_current_records

  for mode in $MODES; do
    echo "===== Running mode: $mode $split target: $target ====="
    bash run_edge_comparison.sh "$mode" "$target" "split"
  done

  rename_records_for_target "$target"
}

run_local_inference() {
  clear_current_records
  echo "===== Running mode: $mode local ====="
  bash run_edge_comparison.sh "http" "127.0.0.1" "local"
  rename_records_for_target "local"
}


cd "$ROOT"

run_split_inference "$CLOUD_NO_TDX_TARGET"
run_split_inference "$CLOUD_TDX_TARGET"

run_local_inference
