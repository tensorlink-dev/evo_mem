#!/usr/bin/env bash
# run_comparison.sh — Head-to-head comparison: Ganglion vs ExpRAG vs History
#
# Usage:
#   export CHUTES_API_KEY="cpk_..."
#   bash scripts/run_comparison.sh            # concurrent (default)
#   PARALLEL=0 bash scripts/run_comparison.sh  # sequential
#
# Optionally override:
#   TASK_LIMIT=50 NUM_STREAMS=3 bash scripts/run_comparison.sh

set -euo pipefail

TASK_LIMIT="${TASK_LIMIT:-100}"
NUM_STREAMS="${NUM_STREAMS:-3}"
PARALLEL="${PARALLEL:-1}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="results/scienceworld_${TIMESTAMP}"

if [ -z "${CHUTES_API_KEY:-}" ]; then
    echo "ERROR: CHUTES_API_KEY is not set."
    echo "  export CHUTES_API_KEY=\"cpk_...\""
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

echo "=========================================="
echo " Evo-Memory Comparison Run"
echo " Output:   ${OUTPUT_BASE}"
echo " Tasks:    ${TASK_LIMIT} | Streams: ${NUM_STREAMS}"
echo " Parallel: $([ "${PARALLEL}" = "1" ] && echo "yes" || echo "no")"
echo "=========================================="

# Helper: run a single agent eval
run_agent() {
    local label="$1" agent="$2" outdir="$3" logfile="$4"
    echo ">>> Starting ${label}..."
    python -m evo_memory.main run \
        --agent "${agent}" \
        --dataset scienceworld \
        --backend chutes \
        --model "moonshotai/Kimi-K2.5-TEE" \
        --task-limit "${TASK_LIMIT}" \
        --num-streams "${NUM_STREAMS}" \
        --output-dir "${outdir}" \
        --seed 42 \
        > "${logfile}" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo ">>> ${label} finished successfully."
    else
        echo ">>> ${label} FAILED (exit code ${rc}). See ${logfile}"
    fi
    return $rc
}

PIDS=()
LABELS=()
LOGS=()

launch() {
    local label="$1" agent="$2" subdir="$3"
    local logfile="${OUTPUT_BASE}/${subdir}.log"
    if [ "${PARALLEL}" = "1" ]; then
        run_agent "${label}" "${agent}" "${OUTPUT_BASE}/${subdir}" "${logfile}" &
        PIDS+=($!)
        LABELS+=("${label}")
        LOGS+=("${logfile}")
    else
        run_agent "${label}" "${agent}" "${OUTPUT_BASE}/${subdir}" "${logfile}"
    fi
}

launch "[1/3] GanglionAgent (hybrid)"   ganglion   ganglion
[ "${PARALLEL}" = "1" ] && sleep 3  # stagger to reduce initial burst
launch "[2/3] ExpRAG baseline"           exprag     exprag
[ "${PARALLEL}" = "1" ] && sleep 3
launch "[3/3] History baseline"          exprecent  history

# Wait for all background jobs and collect exit codes
FAILED=0
if [ "${PARALLEL}" = "1" ]; then
    echo ""
    echo "All 3 agents launched concurrently. Waiting..."
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "FAILED: ${LABELS[$i]} — see ${LOGS[$i]}"
            FAILED=1
        fi
    done
fi

if [ "${FAILED}" -ne 0 ]; then
    echo ""
    echo "WARNING: One or more agents failed. Check logs above."
fi

echo ""
echo "=========================================="
echo " All runs complete."
echo " Analysing results..."
echo "=========================================="

python scripts/compare_results.py "${OUTPUT_BASE}"

echo ""
echo "Done. Results in ${OUTPUT_BASE}/"
