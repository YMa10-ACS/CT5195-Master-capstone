#!/usr/bin/env bash
###
 # @Description: 
 # @Date: 2026-06-07 01:42:34
 # @Author: Yaoquan Ma
### 

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_DIR="$ROOT/../datasets/ai2d/"
MODEL="lusxvr/nanoVLM-230M-8k"
PROMPT="What is this?"
MAX_NEW_TOKENS=300
GENERATIONS=1
MAX_IMAGES=20
HTTP_MODE="${1:-https-aes-gcm}"
CLOUD_TARGET="${2:-${CLOUD_TARGET:-127.0.0.1}}"
CLOUD_HOST="${CLOUD_HOST:-0.0.0.0}"

mkdir -p "$ROOT/record"

CLOUD_LOG="record/distributed_vlm_cloud_${HTTP_MODE}.log"
EDGE_LOG="record/distributed_vlm_edge_${HTTP_MODE}.log"
METRICS_CSV="record/distributed_vlm_metrics_${HTTP_MODE}.csv"

# Kill the cloud service before testing.
kill -9 `sudo lsof -ntP -iTCP:8000 -sTCP:LISTEN`
kill -9 `sudo lsof -ntP -iTCP:8443 -sTCP:LISTEN`
kill -9 `sudo lsof -ntP -iTCP:8444 -sTCP:LISTEN`

cd "$ROOT"

# Lanuch cloud simulation script
python3 ./cloud.py \
  --transport "$HTTP_MODE" \
  --host "$CLOUD_HOST" \
  --weights "$MODEL" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$CLOUD_LOG" &

CLOUD_PID=$!

# Need some time to wait Flash web service starup.
sleep 5

rm -f "$VALIDATION_LOG"

# Lanuch edge simulation script in batch mode. The model is loaded once, then
# each image is processed sequentially.
python3 ./edge.py \
  --transport "$HTTP_MODE" \
  --target "$CLOUD_TARGET" \
  --weights "$MODEL" \
  --image_dir "$IMAGE_DIR" \
  --max_images "$MAX_IMAGES" \
  --generations "$GENERATIONS" \
  --metrics_csv "$METRICS_CSV" \
  | tee "$EDGE_LOG" 2>&1

MATCH_RECORDS=$(grep "matched pair" "$EDGE_LOG")

kill "$CLOUD_PID"


# Calculate matched result
MATCHED_PAIR=0
COMPARED_PAIRS=0

while read -r line; do
  unmatched=$(echo "$line" | sed -E 's/.*unmatch pair \/ matched pair = ([0-9]+)\/([0-9]+).*/\1/')
  matched=$(echo "$line" | sed -E 's/.*unmatch pair \/ matched pair = ([0-9]+)\/([0-9]+).*/\2/')
  compared=$((unmatched + matched))

  MATCHED_PAIR=$((MATCHED_PAIR + matched))
  COMPARED_PAIRS=$((COMPARED_PAIRS + compared))
done < <(echo "$MATCH_RECORDS")

echo "total matched: $MATCHED_PAIR"
echo "total compared: $COMPARED_PAIRS"
