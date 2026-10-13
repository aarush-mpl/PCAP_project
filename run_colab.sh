#!/usr/bin/env bash
# =======================================================================
# Scalable Parallel K-Means (MPI + CUDA) - Google Colab Standalone Runner
# =======================================================================
# Usage in Google Colab:
#   !bash run_colab.sh [N] [D] [K] [Ranks] [InitMethod]
# Example:
#   !bash run_colab.sh 100000 16 10 2 kmeans++
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
RANKS=${4:-2}
INIT=${5:-kmeans++}
MAX_ITERS=${MAX_ITERS:-100}
TOL=${TOL:-1e-4}
SEED=${SEED:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info "Initializing Google Colab execution pipeline..."
log_info "Config: N=$N, D=$D, K=$K, Ranks=$RANKS, Init=$INIT, Seed=$SEED"

log_step "1" "Checking Colab Environment & Dependencies"
mkdir -p data results bin
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    log_warn "No NVIDIA GPU detected by nvidia-smi. Check Colab Runtime -> Change runtime type -> T4 GPU."
fi

if ! command -v mpicxx &> /dev/null; then
    log_info "OpenMPI not found. Installing openmpi-bin libopenmpi-dev..."
    apt-get update -qq && apt-get install -y openmpi-bin libopenmpi-dev
fi

if ! command -v python3 &> /dev/null; then
    apt-get update -qq && apt-get install -y python3 python3-numpy python3-scipy
fi
log_success "Prerequisites verified. Directories data/ and results/ ready."

log_step "2" "Building Targets (serial, mpi, hybrid)"
make clean
make all
make hybrid || log_warn "CUDA hybrid compilation failed; falling back to CPU."

if [[ ! -f "bin/kmeans_serial" || ! -f "bin/kmeans_mpi" ]]; then
    log_error "Build failed! Serial or MPI binary missing."
    exit 1
fi
log_success "Binaries compiled in bin/"

DATASET="data/synth_${N}_d${D}_k${K}.bin"
log_step "3" "Generating Synthetic Clustered Dataset ($DATASET)"
python3 scripts/generate_dataset.py --n "$N" --d "$D" --k "$K" --out "$DATASET" --seed "$SEED"
log_success "Dataset generated."

MPI_FLAGS="--oversubscribe"
if [[ "$(id -u)" -eq 0 ]]; then
    MPI_FLAGS="$MPI_FLAGS --allow-run-as-root"
fi

log_step "4" "Running Serial CPU Baseline Reference"
./bin/kmeans_serial "$DATASET" "$K" "$MAX_ITERS" "$TOL" "$SEED" "results/colab_serial" "$INIT"
log_success "Serial baseline completed."

log_step "5" "Running Distributed MPI CPU Implementation ($RANKS ranks)"
mpirun $MPI_FLAGS -np "$RANKS" ./bin/kmeans_mpi "$DATASET" "$K" "$MAX_ITERS" "$TOL" "$SEED" "results/colab_mpi" "$INIT"
log_success "MPI execution completed."

if [[ -f "bin/kmeans_hybrid" ]]; then
    log_step "6" "Running Hybrid MPI + CUDA GPU Accelerated Binary ($RANKS ranks)"
    mpirun $MPI_FLAGS -np "$RANKS" ./bin/kmeans_hybrid "$DATASET" "$K" "$MAX_ITERS" "$TOL" "$SEED" "results/colab_hybrid" "$INIT"
    log_success "Hybrid CUDA execution completed."
else
    log_warn "bin/kmeans_hybrid not present, skipping Step 6."
fi

log_step "7" "Validating Mathematical Correctness against Serial Reference"
echo "=== Validating MPI vs Serial ==="
python3 scripts/validate_correctness.py results/colab_serial results/colab_mpi
log_success "MPI Mathematical validation passed!"

if [[ -f "bin/kmeans_hybrid" && -f "results/colab_hybrid_metrics.txt" ]]; then
    echo "=== Validating CUDA Hybrid vs Serial ==="
    python3 scripts/validate_correctness.py results/colab_serial results/colab_hybrid
    log_success "Hybrid CUDA Mathematical validation passed!"
fi

echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}      GOOGLE COLAB K-MEANS PIPELINE PASSED!            ${NC}"
echo -e "${GREEN}=======================================================${NC}\n"
