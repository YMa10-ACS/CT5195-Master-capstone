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

CLOUD_LOG="/tmp/distributed_vlm_cloud.log"

# Kill the cloud service before testing.
kill -9 `sudo lsof -ntP -iTCP:8000 -sTCP:LISTEN`

cd "$ROOT"

# Lanuch cloud simulation script
python3 ./cloud.py \
  --weights "$MODEL" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  | tee "$CLOUD_LOG" 2>&1 &

CLOUD_PID=$!

# Need some time to wait Flash web service starup.
sleep 5

rm -f "$VALIDATION_LOG"

COUNT=0
find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read IMAGE; do
  COUNT=$((COUNT + 1))
  if [ "$COUNT" -ge "$MAX_IMAGES" ]; then
    break
  fi
  
  NAME="$(basename "$IMAGE")"
  EDGE_LOG="/tmp/distributed_vlm_edge_${NAME}.log"

  echo "===== Image: $NAME ====="

  # Lanuch edge simulation script
  python3 ./edge.py \
    --weights "$MODEL" \
    --image "$IMAGE" \
  	--generations "$GENERATIONS" \
    | tee "$EDGE_LOG" 2>&1
  echo

  cat $EDGE_LOG | grep "Generation"
  cat $EDGE_LOG | grep "matched_pair"

  MATCH_RECORDS=$(grep "matched pair" "$EDGE_LOG")
done

kill "$CLOUD_PID"


# Calculate matched result
MATCHED_PAIR=0
COMPARED_PAIRS=0

while read -r line; do
  matched=$(echo "$line" | sed -E 's/.*unmatch pair.* ([0-9]+)\/([0-9]+).*/\1/')
  compared=$(echo "$line" | sed -E 's/.*matched pair: ([0-9]+)\/([0-9]+).*/\2/')

  MATCHED_PAIR=$((MATCHED_PAIR + matched))
  COMPARED_PAIRS=$((COMPARED_PAIRS + compared))
done < (echo "$MATCH_RECORDS")

echo "total matched: $MATCHED_PAIR"
echo "total compared: $COMPARED_PAIRS"
