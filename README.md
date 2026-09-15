# Scalable Parallel K-Means — Tirth Patel's Workstream (MPI / Distributed Computing & System Integration)

This is the MPI/distributed-computing portion of the hybrid MPI+CUDA K-Means
project (PS-5). It's built so it **runs and validates completely on its own**
(with a CPU assignment step standing in for the GPU kernel), and drops in
Aarush's CUDA kernel later with a one-line rebuild — no changes to the MPI
control flow required.

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

## Known gaps / next steps

- Rank 0 currently reads the **entire** dataset before scattering it. Fine
  up to the ~1M–100M point / 8–128 dim tier in Section 10's "Synthetic"
  row; for the 256–512-dim "high-dimensional stress test" row or truly
  huge files that don't fit in one node's RAM, switch rank 0's read to
  chunked/streamed I/O (or MPI-IO) feeding `MPI_Scatterv` in batches.
- Centroid initialization is uniform random sampling (deterministic, seeded)
  rather than k-means++; swapping in k-means++ is a drop-in replacement of
  `init_centroids()` if the team wants tighter/faster convergence for the
  paper-quality results.
- `make hybrid` is wired up but expects `src/cuda_assignment_kernel.cu` and
  `$CUDA_HOME` to exist — waiting on Aarush's kernel.
