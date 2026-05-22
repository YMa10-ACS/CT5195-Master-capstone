#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="$ROOT/distributed_vlm/img/1.jpg"
MODEL="lusxvr/nanoVLM-230M-8k"
PROMPT="What is this?"
MAX_NEW_TOKENS=300
GENERATIONS=1

CLOUD_LOG="/tmp/distributed_vlm_cloud.log"
EDGE_LOG="/tmp/distributed_vlm_edge.log"
NANOVLM_LOG="/tmp/nanovlm_generate.log"


cd "$ROOT"

# Lanuch cloud simulation script
python3 distributed_vlm/cloud.py \
  --weights "$MODEL" \
  --prompt "$PROMPT" \
  --generations "$GENERATIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  > "$CLOUD_LOG" 2>&1 &

# Need some time to wait Flash web service starup.
sleep 5

CLOUD_PID=$!

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

echo "===== Distributed VLM result ====="
SIMULATION_RESULT=`grep "Generation" "$EDGE_LOG"`

echo
echo "===== nanoVLM generate.py result ====="
NANOVLM_RESULT=`grep "Generation" "$NANOVLM_LOG"`

if [ "${SIMULATION_RESULT}" == "${NANOVLM_RESULT}" ]; then
  echo "The generate result of simulation system is same as nanoVLM result"
else
  echo "The results are different"
fi

kill "$CLOUD_PID"

