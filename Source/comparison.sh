#!/usr/bin/env bash
###
 # @Description: 
 # @Date: 2026-06-07 01:42:34
 # @Author: Yaoquan Ma
### 
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_DIR="$ROOT/datasets/coco_sample/Data/cat/"
MODEL="lusxvr/nanoVLM-230M-8k"
PROMPT="What is this?"
MAX_NEW_TOKENS=300
GENERATIONS=1
MAX_IMAGES=10000

CLOUD_LOG="/tmp/distributed_vlm_cloud.log"
VALIDATION_LOG="/tmp/validate_split.log"



cd "$ROOT"

# Lanuch cloud simulation script
python3 distributed_vlm/cloud.py \
  --weights "$MODEL" \
  --prompt "$PROMPT" \
  --generations "$GENERATIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  > "$CLOUD_LOG" 2>&1 &

CLOUD_PID=$!

# Need some time to wait Flash web service starup.
sleep 5

rm -f "$VALIDATION_LOG"

COUNT=0
find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read IMAGE; do
  COUNT=$((COUNT + 1))
  if [ "$COUNT" -gt "$MAX_IMAGES" ]; then
    break
  fi
  
  NAME="$(basename "$IMAGE")"
  EDGE_LOG="/tmp/distributed_vlm_edge_${NAME}.log"
  NANOVLM_LOG="/tmp/nanovlm_generate_${NAME}.log"

  echo "===== Image: $NAME ====="

  # Lanuch edge simulation script
  python3 distributed_vlm/edge.py \
    --weights "$MODEL" \
    --image "$IMAGE" \
    > "$EDGE_LOG" 2>&1

  # Runs the generate script to generate nanoVLM text
  python3 nanoVLM/generate.py \
    --hf_model "$MODEL" \
    --image "$IMAGE" \
    --prompt "$PROMPT" \
    --generations "$GENERATIONS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    > "$NANOVLM_LOG" 2>&1

  python3 validate_split.py \
    --edge-log "$EDGE_LOG" \
    --nanovlm-log "$NANOVLM_LOG" \
    --weights "$MODEL" \
    --image "$NAME" | tee -a "$VALIDATION_LOG"

  echo
done

kill "$CLOUD_PID"


# Calculate matched result
MATCHED_PAIR=0
COMPARED_PAIRS=0

while read -r line; do
  matched=$(echo "$line" | sed -E 's/.*semantic matches: ([0-9]+)\/([0-9]+).*/\1/')
  compared=$(echo "$line" | sed -E 's/.*semantic matches: ([0-9]+)\/([0-9]+).*/\2/')

  MATCHED_PAIR=$((MATCHED_PAIR + matched))
  COMPARED_PAIRS=$((COMPARED_PAIRS + compared))
done < <(grep "semantic matches" "$VALIDATION_LOG")

echo "total matched: $MATCHED_PAIR"
echo "total compared: $COMPARED_PAIRS"
