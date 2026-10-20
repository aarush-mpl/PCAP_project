#!/usr/bin/env bash
# =======================================================================
# Scalable Parallel K-Means (MPI + CUDA) - Linux / WSL Execution Script
# =======================================================================
# Runs the entire build, dataset generation, serial baseline, parallel MPI,
# hybrid MPI+CUDA acceleration, correctness validation, and scaling sweep.
#
# Usage:
#   ./run_project.sh                   (defaults: N=100000, D=16, K=10, P=4, Init=kmeans++)
#   ./run_project.sh 500000 32 20 8    (custom: N=500k, D=32, K=20, P=8)
# =======================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step() { echo -e "\n${CYAN}[STEP $1] ${YELLOW}$2${NC}"; }
log_success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
log_info() { echo -e "${CYAN}[INFO] $1${NC}"; }
log_warn() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
log_error() { echo -e "${RED}[ERROR] $1${NC}"; }

N=${1:-100000}
D=${2:-16}
K=${3:-10}
RANKS=${4:-4}
INIT=${5:-kmeans++}
MAX_ITERS=${MAX_ITERS:-100}
TOL=${TOL:-1e-4}
SEED=${SEED:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info "Starting Parallel K-Means Execution Pipeline in WSL/Linux..."
log_info "Parameters: N=$N, D=$D, K=$K, Ranks=$RANKS, Init=$INIT, Seed=$SEED"

log_step "1" "Preparing Directories & Checking Dependencies"
mkdir -p data results bin
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

if ! command -v mpicxx &> /dev/null; then
    log_error "mpicxx not found! Install with: sudo apt-get install -y openmpi-bin libopenmpi-dev"
    exit 1
fi
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found! Install with: sudo apt-get install -y python3 python3-numpy python3-scipy"
    exit 1
fi
log_success "Prerequisites verified. Output directories data/ and results/ ready."

log_step "2" "Building Project Targets (serial, mpi, hybrid)"
make clean
make all
make hybrid || log_warn "CUDA compilation skipped or failed; continuing with CPU fallback."

if [[ ! -f "bin/kmeans_serial" || ! -f "bin/kmeans_mpi" ]]; then
    log_error "Build failed! Serial or MPI binary missing."
    exit 1
fi
log_success "Binaries compiled successfully in bin/"

DATASET="data/synth_${N}_d${D}_k${K}.bin"
log_step "3" "Generating Benchmark Dataset ($DATASET)"
python3 scripts/generate_dataset.py --n "$N" --d "$D" --k "$K" --out "$DATASET" --seed "$SEED"
log_success "Dataset generated successfully."

log_step "4" "Running Serial CPU Baseline Reference"
./bin/kmeans_serial "$DATASET" "$K" "$MAX_ITERS" "$TOL" "$SEED" "results/demo_serial" "$INIT"
log_success "Serial baseline run completed."

log_step "5" "Running Distributed MPI Version ($RANKS ranks)"
MPI_FLAGS=""
if [[ "$(id -u)" -eq 0 ]]; then
    MPI_FLAGS="--allow-run-as-root"
fi
mpirun $MPI_FLAGS -np "$RANKS" ./bin/kmeans_mpi "$DATASET" "$K" "$MAX_ITERS" "$TOL" "$SEED" "results/demo_mpi" "$INIT"
log_success "MPI execution completed across $RANKS ranks."

log_step "6" "Running Hybrid MPI + CUDA Binary"
if [[ -f "bin/kmeans_hybrid" ]]; then
    mpirun $MPI_FLAGS -np "$RANKS" ./bin/kmeans_hybrid "$DATASET" "$K" "$MAX_ITERS" "$TOL" "$SEED" "results/demo_hybrid" "$INIT"
    log_success "Hybrid execution completed."
else
    log_warn "bin/kmeans_hybrid not present, skipping Step 6."
fi

log_step "7" "Validating MPI & Hybrid Correctness against Serial Reference"
python3 scripts/validate_correctness.py results/demo_serial results/demo_mpi
log_success "MPI Mathematical validation passed!"

if [[ -f "bin/kmeans_hybrid" && -f "results/demo_hybrid_metrics.txt" ]]; then
    python3 scripts/validate_correctness.py results/demo_serial results/demo_hybrid
    log_success "Hybrid Mathematical validation passed!"
fi

log_step "8" "Running Strong-Scaling Experiment Sweep"
./scripts/run_scaling_experiments.sh strong "$DATASET" "$K"
log_success "Scaling sweep completed! Results saved in results/"

echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}   PARALLEL K-MEANS WSL/LINUX DEMO PIPELINE PASSED!   ${NC}"
echo -e "${GREEN}=======================================================${NC}\n"
