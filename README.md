# Scalable Parallel K-Means — Team Workstreams (MPI / Distributed Computing & GPU Integration)

This hybrid MPI+CUDA K-Means project is split into complementary workstreams:

- Tirth Patel — MPI/distributed-computing layer, dataset partitioning,
  synchronization, and scalability benchmarking.
- Dispo — CUDA/GPU-side assignment kernel, device-side optimization, and
  performance validation against the serial and MPI baselines.

This repository contains Tirth's MPI/distributed-computing portion of the
project (PS-5). It's built so it **runs and validates completely on its own**
(with a CPU assignment step standing in for the GPU kernel), and is designed to
accept Dispo's CUDA implementation later with a one-line rebuild — no changes
to the MPI control flow required.

## Team contribution summary

The project combines a distributed CPU/MPI implementation with a GPU-accelerated
assignment kernel. Tirth focuses on the communication and scaling model, while
Dispo contributes the CUDA workstream responsible for the point-to-centroid
assignment computation, kernel optimization, and GPU-side performance checks.

## What's here

```
include/kmeans_common.h     Shared dataset I/O, partition planning, K-Means math
                             (used by every binary, so serial/MPI/hybrid all
                             agree on the exact same math and file formats)
src/kmeans_serial.cpp       Serial CPU reference baseline (Section 13)
src/kmeans_mpi.cpp          Tirth's core deliverable: MPI init/rank mgmt,
                             dataset partitioning, MPI_Scatterv distribution,
                             MPI_Allreduce centroid update, timing instrumentation,
                             and the CUDA integration hook
scripts/generate_dataset.py Synthetic clustered dataset generator (Section 10)
scripts/prepare_real_data.py Real-world dataset pipeline: MNIST / KDD Cup 1999 /
                             US Census 1990 -> the same .bin format (Section 10)
scripts/run_scaling_experiments.sh  Strong/weak scaling sweep harness (Section 12)
scripts/validate_correctness.py     Permutation-invariant serial-vs-parallel
                             correctness check (Section 15)
Makefile                    Build targets: serial, mpi, hybrid
```

## Build

```bash
# Serial baseline (no MPI needed)
make serial

# MPI version — requires an MPI toolchain (OpenMPI or MPICH), e.g.:
#   Ubuntu/Debian: sudo apt-get install libopenmpi-dev openmpi-bin
#   Then:
make mpi
```

`make mpi` invokes `mpicxx` directly, so any standards-compliant MPI
implementation on your `PATH` works.

## Run

```bash
# 1. Generate a test dataset (needs numpy)
python3 scripts/generate_dataset.py --n 1000000 --d 16 --k 20 \
    --out data/synth_1M_d16_k20.bin --seed 42

# 2. Serial reference
./bin/kmeans_serial data/synth_1M_d16_k20.bin 20 100 1e-4 42 results/serial

# 3. MPI version, e.g. 4 ranks
mpirun -np 4 ./bin/kmeans_mpi data/synth_1M_d16_k20.bin 20 100 1e-4 42 results/mpi

# 4. Validate MPI output against the serial reference
python3 scripts/validate_correctness.py results/serial results/mpi
```

Each run writes `<prefix>_centroids.txt`, `<prefix>_assignments.txt`, and
`<prefix>_metrics.txt` (iterations, SSE, and — for the MPI binary — a
compute/communication time breakdown per Section 12's "Communication
Overhead" metric).

## Real-world data (Section 10: US Census, KDD Cup, MNIST)

`scripts/prepare_real_data.py` downloads and converts real datasets into
the same `.bin` format everything else reads, using standard, citable
loaders rather than scraped mirrors:

| `--dataset`      | Source                                                    | Typical size (unfiltered)          |
|-------------------|-----------------------------------------------------------|-------------------------------------|
| `mnist`           | `sklearn.datasets.fetch_openml('mnist_784')` (OpenML)     | 70,000 × 784                        |
| `kddcup99`        | `sklearn.datasets.fetch_kddcup99()` (UCI KDD Archive)      | 494,021 × ~118 (10%) / 4.9M × ~118 (full) |
| `uscensus1990`    | `ucimlrepo.fetch_ucirepo(id=116)` (official UCI client)    | 2,458,285 × 68                      |

**This sandbox has no internet access, so these downloads must be run on
your own machine** (e.g. the VS Code integrated terminal), where you have
connectivity:

```bash
pip install scikit-learn pandas ucimlrepo

# MNIST (~55MB download, cached after first run)
python3 scripts/prepare_real_data.py --dataset mnist --out data/real_mnist.bin

# KDD Cup 1999, standard 10% subset
python3 scripts/prepare_real_data.py --dataset kddcup99 --percent10 \
    --out data/real_kddcup99_10pct.bin

# US Census 1990 -- 2.4M rows; cap it while iterating locally, then
# drop --max-n for the full-scale run that matches the project's
# "big data" aim
python3 scripts/prepare_real_data.py --dataset uscensus1990 --max-n 500000 \
    --out data/real_census_500k.bin
```

Each run also writes a `<name>_report.txt` alongside the `.bin` file
recording the exact source, citation, shape, and preprocessing applied
(one-hot encoding for KDD Cup's categorical columns, standardization,
any subsampling) — paste that straight into the report's "Real-world
validation" section for reproducibility.

Then run exactly like the synthetic data:

```bash
./bin/kmeans_serial   data/real_mnist.bin 10 100 1e-4 42 results/mnist_serial
mpirun -np 4 ./bin/kmeans_mpi data/real_mnist.bin 10 100 1e-4 42 results/mnist_mpi
python3 scripts/validate_correctness.py results/mnist_serial results/mnist_mpi
```

**Preprocessing notes**
- All three loaders standardize features (zero mean, unit variance) by
  default (`--no-standardize` to skip) — squared-Euclidean K-Means is
  scale-sensitive, and these datasets mix wildly different feature
  scales (pixel intensities, byte counts, one-hot flags, census codes).
- KDD Cup 1999's `protocol_type`/`service`/`flag` columns are one-hot
  encoded; the `labels` (attack type) column is dropped since this is an
  unsupervised clustering task, not classification.
- US Census 1990's fields are already integer-coded categorical/ordinal
  values from the source (not continuous measurements). They're used
  directly as numeric features here, which is standard practice for
  scalability benchmarking but means treat resulting clusters as a
  throughput/scaling benchmark first, not a validated demographic
  segmentation — call this out explicitly in the report if you present
  cluster interpretations, not just runtime numbers.
- I have validated `prepare_real_data.py`'s full pipeline (each loader's
  data-shaping logic, standardization, `.bin` writing, and the resulting
  file being read correctly by `kmeans_serial`) using synthetic stand-ins
  for the three network calls, since this environment has no outbound
  internet access. The network calls themselves (`fetch_openml`,
  `fetch_kddcup99`, `fetch_ucirepo`) are well-established library
  functions — run it once on your machine and sanity-check the printed
  N/D against the table above.

## Scaling experiments

```bash
make mpi
./scripts/run_scaling_experiments.sh strong data/synth_10M_d16_k20.bin 20
# generates results/scaling_strong_<timestamp>.csv with one row per rank count
```

For weak scaling, pre-generate one dataset per rank count so the
per-rank problem size stays constant as rank count grows (e.g.
`base_r1.bin`, `base_r2.bin`, `base_r4.bin`, ... each `n = base_n * ranks`),
then:

```bash
./scripts/run_scaling_experiments.sh weak data/synth_base_d16_k20 20
```

## Design notes (what the MPI layer does)

1. **Rank management**: `MPI_Init` / `MPI_Comm_rank` / `MPI_Comm_size` at
   startup; every subsequent step is written in terms of `world_rank` /
   `world_size` so it works unchanged from 1 rank up to however many the
   cluster has.
2. **Partitioning**: `plan_partition()` in `kmeans_common.h` splits `N`
   points as evenly as possible across ranks (remainder points go one each
   to the first `N % P` ranks), producing the counts/displacements
   `MPI_Scatterv`/`MPI_Gatherv` need for uneven chunks.
3. **Distribution**: only rank 0 ever holds the full dataset in memory —
   `MPI_Scatterv` ships each rank its own contiguous `local_N * D` float
   buffer, which doubles as the exact layout a CUDA kernel would want to
   `cudaMemcpy` to the device.
4. **Global centroid update**: after each rank computes its local partial
   cluster sums/counts (CPU today, GPU once integrated), `MPI_Allreduce`
   (not `MPI_Reduce`) combines them across all ranks, because every rank
   needs the updated global centroids to start its *next* local iteration —
   this is the textbook use case `MPI_Allreduce` is for.
5. **Timing**: computation time (the assignment step) and communication
   time (the `Allreduce` calls) are measured separately with `MPI_Wtime`
   and reported both per-rank (min/max, to catch load imbalance) and
   averaged, satisfying the "record communication time separately" goal in
   Section 9.2.
6. **Correctness**: global assigned-point count is checked against `N`
   after every reduction; `scripts/validate_correctness.py` then compares
   a full run's SSE and (permutation-matched) centroids against the serial
   reference within a stated tolerance, per Section 15.

## CUDA integration contract (for Aarush's module)

`src/kmeans_mpi.cpp` calls a local `do_assignment_step()` wrapper each
iteration. Today (no `USE_CUDA_KERNEL` define) that calls
`assign_points_cpu()` from `kmeans_common.h`. To integrate the CUDA kernel:

1. Implement, in a `.cu` file, an `extern "C"` function with this exact
   signature:
   ```c
   double launch_assignment_kernel(const float* local_points, int local_N, int D,
                                    const float* centroids, int K,
                                    int* out_assign,
                                    double* out_local_sums,
                                    int* out_local_cnts);
   ```
   It owns the H2D copy of `local_points`/`centroids`, the kernel launch,
   the D2H copy of `out_assign`, and accumulating `out_local_sums`/
   `out_local_cnts`/the returned SSE — i.e. the same contract
   `assign_points_cpu()` already fulfills.
2. Build with `make hybrid` (compiles the `.cu` file with `nvcc`, then
   links it into `kmeans_mpi.cpp` compiled with `-DUSE_CUDA_KERNEL`).

No other change to the MPI control flow, partitioning, or the
`MPI_Allreduce` centroid update is needed — that's the whole point of the
hook.

## Features Completed & Implementation Status

- **CUDA Assignment Kernel (`src/cuda_assignment_kernel.cu`)**: Implemented the GPU-accelerated assignment kernel with CUDA grid execution, atomic reductions for cluster sums/counts, and host fallback when built without `nvcc`.
- **K-Means++ Centroid Initialization**: Implemented `init_centroids_kmeans_pp()` in `include/kmeans_common.h` for probabilistic $D(x)^2$ centroid sampling, providing tighter and faster convergence. Selectable via `init=kmeans++`.
- **Streamed Dataset Scatter**: Added chunked dataset reading support in `src/kmeans_mpi.cpp` and `include/kmeans_common.h`.
- **Directory Structure**: Fully organized into `include/`, `src/`, `scripts/`, `bin/`, `data/`, and `results/`.

## Running & Testing in WSL (Ubuntu / Kali Linux)

### 1. Prerequisites (WSL Kali / Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y build-essential mpich libopenmpi-dev python3-numpy python3-scipy
```

### 2. Build All Targets
```bash
# Serial CPU reference baseline
make serial

# Distributed MPI baseline
make mpi

# Hybrid MPI + CUDA GPU binary (requires nvcc / CUDA toolkit)
make hybrid
```

### 3. Execution & Verification Pipeline
```bash
# Generate synthetic benchmark dataset
python3 scripts/generate_dataset.py --n 100000 --d 16 --k 10 --out data/synth_100k.bin

# Run Serial baseline (using K-Means++)
./bin/kmeans_serial data/synth_100k.bin 10 100 1e-4 42 results/serial_run kmeans++

# Run MPI version with 4 ranks (using K-Means++)
mpirun -np 4 ./bin/kmeans_mpi data/synth_100k.bin 10 100 1e-4 42 results/mpi_run kmeans++

# Run Hybrid MPI+CUDA version (using K-Means++)
mpirun -np 4 ./bin/kmeans_hybrid data/synth_100k.bin 10 100 1e-4 42 results/hybrid_run kmeans++

# Validate MPI & Hybrid output against Serial baseline
python3 scripts/validate_correctness.py results/serial_run results/mpi_run
python3 scripts/validate_correctness.py results/serial_run results/hybrid_run
```

