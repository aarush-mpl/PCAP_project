#!/usr/bin/env bash
# -----------------------------------------------------------------------
# run_scaling_experiments.sh
#
# Runs the strong-scaling and weak-scaling sweeps that feed Section 12's
# runtime/speedup/efficiency plots. Appends one CSV row per run so
# results survive a crashed sweep and can be reloaded/plotted anytime.
#
# Owner: Tirth Patel (MPI experiment scripts + reproducibility configs)
#
# Usage:
#   ./scripts/run_scaling_experiments.sh strong data/synth_10M_d16_k20.bin 20
#   ./scripts/run_scaling_experiments.sh weak    data/synth_base_d16_k20   20
#
# Requires: bin/kmeans_mpi already built (make mpi), mpirun on PATH.
# -----------------------------------------------------------------------
set -euo pipefail

MODE=${1:-strong}          # strong | weak
DATASET_OR_PREFIX=${2:?dataset path (strong) or dataset-prefix (weak) required}
K=${3:?K required}
MAX_ITERS=${MAX_ITERS:-100}
TOL=${TOL:-1e-4}
SEED=${SEED:-42}
RANKS=${RANKS:-"1 2 4 8 16"}
BIN=${BIN:-bin/kmeans_mpi}
RESULTS=${RESULTS:-results}
mkdir -p "$RESULTS"

CSV="$RESULTS/scaling_${MODE}_$(date +%Y%m%d_%H%M%S).csv"
echo "mode,ranks,dataset,N,D,K,iterations,SSE,total_time_s,compute_avg_s,comm_avg_s,comm_overhead_fraction" > "$CSV"

run_one() {
    local ranks=$1
    local dataset=$2
    local out_prefix="$RESULTS/${MODE}_r${ranks}"

    echo ">>> [$MODE] ranks=$ranks dataset=$dataset"
    mpirun -np "$ranks" "$BIN" "$dataset" "$K" "$MAX_ITERS" "$TOL" "$SEED" "$out_prefix" \
        > "${out_prefix}.log" 2>&1

    # Parse the metrics file this run wrote and append a CSV row.
    local metrics="${out_prefix}_metrics.txt"
    if [[ -f "$metrics" ]]; then
        local N D iters sse total compute comm frac
        N=$(awk '/^N /{print $2}' "$metrics")
        D=$(awk '/^D /{print $2}' "$metrics")
        iters=$(awk '/^iterations /{print $2}' "$metrics")
        sse=$(awk '/^SSE /{print $2}' "$metrics")
        total=$(awk '/^total_time_s /{print $2}' "$metrics")
        compute=$(awk '/^compute_time_avg_s /{print $2}' "$metrics")
        comm=$(awk '/^comm_time_avg_s /{print $2}' "$metrics")
        frac=$(awk '/^comm_overhead_fraction /{print $2}' "$metrics")
        echo "$MODE,$ranks,$dataset,$N,$D,$K,$iters,$sse,$total,$compute,$comm,$frac" >> "$CSV"
    else
        echo "!!! run failed for ranks=$ranks, see ${out_prefix}.log" >&2
    fi
}

if [[ "$MODE" == "strong" ]]; then
    # Fixed dataset size, increasing rank count.
    DATASET="$DATASET_OR_PREFIX"
    for r in $RANKS; do
        run_one "$r" "$DATASET"
    done

elif [[ "$MODE" == "weak" ]]; then
    # Dataset size grows proportionally with rank count. Expects
    # pre-generated files named "<prefix>_r<ranks>.bin" (generate with
    # generate_dataset.py at the matching per-rank point count ahead of
    # time, e.g. base_points_per_rank * ranks).
    PREFIX="$DATASET_OR_PREFIX"
    for r in $RANKS; do
        run_one "$r" "${PREFIX}_r${r}.bin"
    done

else
    echo "Unknown mode: $MODE (expected 'strong' or 'weak')" >&2
    exit 1
fi

echo "Results written to $CSV"
