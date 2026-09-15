// kmeans_common.h
// -----------------------------------------------------------------------
// Shared utilities for the Scalable Parallel K-Means project (PS-5).
// Used by BOTH the serial reference implementation and the MPI+CUDA
// implementation, so the on-disk dataset format and the K-Means math are
// identical everywhere -> correctness comparisons are apples-to-apples.
//
// Owner: Tirth Patel (MPI / Distributed Computing & System Integration)
// -----------------------------------------------------------------------
#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

// -----------------------------------------------------------------------
// Binary dataset format (kept simple + fast to read for million-point
// datasets; avoids CSV parsing overhead on the hot path):
//
//   int32_t magic       = 0x4B4D4E53  ("KMNS")
//   int32_t N            (number of points)
//   int32_t D            (dimensionality)
//   float32 data[N * D]  (row-major: point i, dim d -> data[i*D + d])
//
// scripts/generate_dataset.py writes exactly this format.
// -----------------------------------------------------------------------
static const int32_t KMEANS_MAGIC = 0x4B4D4E53;

struct Dataset {
    int N = 0;   // number of points
    int D = 0;   // dimensionality
    std::vector<float> data; // size N*D, row-major, contiguous (GPU-friendly)
};

// Read the whole dataset (used by: serial baseline, and rank 0 in the MPI
// version before it scatters partitions out to the other ranks).
inline Dataset read_dataset_full(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open dataset file: " + path);

    int32_t magic = 0, N = 0, D = 0;
    if (std::fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != KMEANS_MAGIC)
        throw std::runtime_error("Bad dataset file (magic mismatch): " + path);
    if (std::fread(&N, sizeof(int32_t), 1, f) != 1) throw std::runtime_error("Bad header (N)");
    if (std::fread(&D, sizeof(int32_t), 1, f) != 1) throw std::runtime_error("Bad header (D)");
    if (N <= 0 || D <= 0) throw std::runtime_error("Invalid N/D in dataset header");

    Dataset ds;
    ds.N = N;
    ds.D = D;
    ds.data.resize((size_t)N * (size_t)D);
    size_t got = std::fread(ds.data.data(), sizeof(float), (size_t)N * (size_t)D, f);
    std::fclose(f);
    if (got != (size_t)N * (size_t)D)
        throw std::runtime_error("Dataset file truncated: " + path);
    return ds;
}

// Read only the header (N, D) — used by MPI rank 0 to plan the partition
// before deciding how much data to stream to each rank.
inline void read_dataset_header(const std::string& path, int& N, int& D) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open dataset file: " + path);
    int32_t magic = 0, n32 = 0, d32 = 0;
    if (std::fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != KMEANS_MAGIC)
        throw std::runtime_error("Bad dataset file (magic mismatch): " + path);
    std::fread(&n32, sizeof(int32_t), 1, f);
    std::fread(&d32, sizeof(int32_t), 1, f);
    std::fclose(f);
    N = n32; D = d32;
}

// -----------------------------------------------------------------------
// Partition planning: split N points as evenly as possible across
// `world_size` ranks. Returns per-rank counts (in POINTS, not floats) and
// displacements (in POINTS) suitable for MPI_Scatterv/MPI_Gatherv once
// multiplied by D. Kept as plain functions (no MPI type) so it is
// trivially unit-testable without an MPI runtime.
// -----------------------------------------------------------------------
inline void plan_partition(int N, int world_size,
                            std::vector<int>& counts,
                            std::vector<int>& displs) {
    counts.assign(world_size, 0);
    displs.assign(world_size, 0);
    int base = N / world_size;
    int rem  = N % world_size;
    int offset = 0;
    for (int r = 0; r < world_size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0); // first `rem` ranks get one extra point
        displs[r] = offset;
        offset += counts[r];
    }
}

// -----------------------------------------------------------------------
// Centroid initialization: deterministic (seeded) uniform sampling of K
// distinct points from the dataset. Kept deterministic so every rank can
// independently reproduce the SAME initial centroids given the same seed
// (avoids having to broadcast the initial centroids at iteration 0 if a
// process later needs to recompute independently, and makes correctness
// comparisons against the serial baseline reproducible).
// -----------------------------------------------------------------------
inline std::vector<float> init_centroids(const Dataset& ds, int K, unsigned seed) {
    if (K > ds.N) throw std::runtime_error("K cannot exceed number of points");
    std::vector<int> idx(ds.N);
    for (int i = 0; i < ds.N; ++i) idx[i] = i;
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<float> centroids((size_t)K * ds.D);
    for (int k = 0; k < K; ++k) {
        const float* src = &ds.data[(size_t)idx[k] * ds.D];
        std::memcpy(&centroids[(size_t)k * ds.D], src, sizeof(float) * ds.D);
    }
    return centroids;
}

// -----------------------------------------------------------------------
// CPU reference assignment step (squared Euclidean distance).
// This is the function signature both:
//   (a) the serial baseline, and
//   (b) the MPI version's CPU fallback / correctness cross-check
// use. In the hybrid MPI+CUDA build, Aarush's CUDA kernel replaces the
// CALL to this function on each rank (same inputs/outputs — see the
// "CUDA INTEGRATION HOOK" comment in src/kmeans_mpi.cpp) without changing
// anything else in the MPI control flow.
//
// local_points   : local_N * D floats, row-major
// centroids      : K * D floats, row-major
// out_assign     : local_N ints, nearest centroid index per local point
// out_local_sums : K * D floats, accumulated (zeroed by caller each iter)
// out_local_cnts : K ints, accumulated (zeroed by caller each iter)
// returns local sum of squared distances (partial SSE contribution)
// -----------------------------------------------------------------------
inline double assign_points_cpu(const float* local_points, int local_N, int D,
                                 const float* centroids, int K,
                                 int* out_assign,
                                 double* out_local_sums,
                                 int* out_local_cnts) {
    double local_sse = 0.0;
    for (int i = 0; i < local_N; ++i) {
        const float* p = local_points + (size_t)i * D;
        int best_k = 0;
        double best_dist = 1e300;
        for (int k = 0; k < K; ++k) {
            const float* c = centroids + (size_t)k * D;
            double dist = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = (double)p[d] - (double)c[d];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_k = k; }
        }
        out_assign[i] = best_k;
        local_sse += best_dist;
        out_local_cnts[best_k] += 1;
        double* sum_row = out_local_sums + (size_t)best_k * D;
        for (int d = 0; d < D; ++d) sum_row[d] += p[d];
    }
    return local_sse;
}

// Combine global sums/counts (already reduced, e.g. via MPI_Allreduce)
// into new centroids. Clusters with zero assigned points keep their old
// centroid (documented degenerate-cluster handling).
inline int update_centroids(const double* global_sums, const int* global_cnts,
                             int K, int D, std::vector<float>& centroids,
                             double* out_max_shift) {
    int empty_clusters = 0;
    double max_shift_sq = 0.0;
    for (int k = 0; k < K; ++k) {
        if (global_cnts[k] == 0) { empty_clusters++; continue; }
        double shift_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double new_val = global_sums[(size_t)k * D + d] / (double)global_cnts[k];
            double diff = new_val - (double)centroids[(size_t)k * D + d];
            shift_sq += diff * diff;
            centroids[(size_t)k * D + d] = (float)new_val;
        }
        if (shift_sq > max_shift_sq) max_shift_sq = shift_sq;
    }
    if (out_max_shift) *out_max_shift = std::sqrt(max_shift_sq);
    return empty_clusters;
}

// Write centroids + run metadata to a simple text file for downstream
// analysis (plots/tables) and cross-run correctness comparison.
inline void write_centroids(const std::string& path, const std::vector<float>& centroids,
                             int K, int D) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("Cannot open output file: " + path);
    std::fprintf(f, "%d %d\n", K, D);
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d)
            std::fprintf(f, "%.6f%s", centroids[(size_t)k * D + d], d + 1 < D ? " " : "\n");
    }
    std::fclose(f);
}

inline void write_assignments(const std::string& path, const std::vector<int>& assign) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("Cannot open output file: " + path);
    for (size_t i = 0; i < assign.size(); ++i) std::fprintf(f, "%d\n", assign[i]);
    std::fclose(f);
}

#endif // KMEANS_COMMON_H
