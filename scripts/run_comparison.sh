#!/usr/bin/env bash
# run_comparison.sh — Head-to-head comparison: Ganglion vs ExpRAG vs History vs Zero-shot
#
# Usage:
#   export CHUTES_API_KEY="cpk_..."
#   bash scripts/run_comparison.sh
#
# Optionally override:
#   TASK_LIMIT=50 NUM_STREAMS=3 bash scripts/run_comparison.sh

set -euo pipefail

TASK_LIMIT="${TASK_LIMIT:-100}"
NUM_STREAMS="${NUM_STREAMS:-3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="results/comparison_${TIMESTAMP}"

if [ -z "${CHUTES_API_KEY:-}" ]; then
    echo "ERROR: CHUTES_API_KEY is not set."
    echo "  export CHUTES_API_KEY=\"cpk_...\""
    exit 1
fi

echo "=========================================="
echo " Evo-Memory Comparison Run"
echo " Output: ${OUTPUT_BASE}"
echo " Tasks:  ${TASK_LIMIT} | Streams: ${NUM_STREAMS}"
echo "=========================================="

# --- 0. Zero-shot Baseline (no memory) ---
echo ""
echo ">>> [1/4] Running Zero-shot baseline..."
python -m evo_memory.main run \
    --agent zeroshot \
    --dataset mmlu_pro \
    --backend chutes \
    --model "deepseek-ai/DeepSeek-R1" \
    --task-limit "${TASK_LIMIT}" \
    --num-streams "${NUM_STREAMS}" \
    --output-dir "${OUTPUT_BASE}/zeroshot" \
    --seed 42

# --- 1. Ganglion Agent (hybrid mode) ---
echo ""
echo ">>> [2/4] Running GanglionAgent (hybrid)..."
python -m evo_memory.main run \
    --agent ganglion \
    --dataset mmlu_pro \
    --backend chutes \
    --model "deepseek-ai/DeepSeek-R1" \
    --task-limit "${TASK_LIMIT}" \
    --num-streams "${NUM_STREAMS}" \
    --output-dir "${OUTPUT_BASE}/ganglion" \
    --seed 42

# --- 2. ExpRAG Baseline ---
echo ""
echo ">>> [3/4] Running ExpRAG baseline..."
python -m evo_memory.main run \
    --agent exprag \
    --dataset mmlu_pro \
    --backend chutes \
    --model "deepseek-ai/DeepSeek-R1" \
    --task-limit "${TASK_LIMIT}" \
    --num-streams "${NUM_STREAMS}" \
    --output-dir "${OUTPUT_BASE}/exprag" \
    --seed 42

# --- 3. History Baseline (ExpRecent) ---
echo ""
echo ">>> [4/4] Running History baseline (ExpRecent)..."
python -m evo_memory.main run \
    --agent exprecent \
    --dataset mmlu_pro \
    --backend chutes \
    --model "deepseek-ai/DeepSeek-R1" \
    --task-limit "${TASK_LIMIT}" \
    --num-streams "${NUM_STREAMS}" \
    --output-dir "${OUTPUT_BASE}/history" \
    --seed 42

echo ""
echo "=========================================="
echo " All runs complete."
echo " Analysing results..."
echo "=========================================="

python scripts/compare_results.py "${OUTPUT_BASE}"

echo ""
echo "Done. Results in ${OUTPUT_BASE}/"
