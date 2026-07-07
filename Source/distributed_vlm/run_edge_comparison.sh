#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_DIR="${IMAGE_DIR:-$ROOT/../datasets/ai2d/}"
MODEL="${MODEL:-lusxvr/nanoVLM-230M-8k}"
PROMPT="${PROMPT:-What is this?}"
GENERATIONS="${GENERATIONS:-1}"
MAX_IMAGES="${MAX_IMAGES:-1}"
HTTP_MODE="${1:-${HTTP_MODE:-https-aes-gcm}}"
CLOUD_TARGET="${2:-${CLOUD_TARGET:-127.0.0.1}}"
SPLIT_MODE="${3:-both}"

mkdir -p "$ROOT/record"

EDGE_LOG="$ROOT/record/distributed_vlm_edge_${HTTP_MODE}.log"
METRICS_CSV="$ROOT/record/distributed_vlm_metrics_${HTTP_MODE}.csv"
THROUGHPUT_CSV="$ROOT/record/distributed_vlm_throughput_${HTTP_MODE}.csv"

cd "$ROOT"

python3 ./edge.py \
  --split "$SPLIT_MODE" \
  --transport "$HTTP_MODE" \
  --target "$CLOUD_TARGET" \
  --weights "$MODEL" \
  --image_dir "$IMAGE_DIR" \
  --max_images "$MAX_IMAGES" \
  --generations "$GENERATIONS" \
  --metrics_csv "$METRICS_CSV" \
  --throughput_csv "$THROUGHPUT_CSV" \
  | tee "$EDGE_LOG" 2>&1

MATCH_RECORDS=$(grep "matched pair" "$EDGE_LOG")

MATCHED_PAIR=0
COMPARED_PAIRS=0

if [ -n "$MATCH_RECORDS" ]; then
  while read -r line; do
    unmatched=$(echo "$line" | sed -E 's/.*unmatch pair \/ matched pair = ([0-9]+)\/([0-9]+).*/\1/')
    matched=$(echo "$line" | sed -E 's/.*unmatch pair \/ matched pair = ([0-9]+)\/([0-9]+).*/\2/')
    compared=$((unmatched + matched))

    MATCHED_PAIR=$((MATCHED_PAIR + matched))
    COMPARED_PAIRS=$((COMPARED_PAIRS + compared))
  done < <(echo "$MATCH_RECORDS")
fi

echo "total matched: $MATCHED_PAIR"
echo "total compared: $COMPARED_PAIRS"
