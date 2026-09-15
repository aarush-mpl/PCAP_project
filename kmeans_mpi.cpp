// kmeans_mpi.cpp
// -----------------------------------------------------------------------
// MPI layer for the hybrid MPI + CUDA K-Means system (PS-5).
//
// Owner: Tirth Patel — MPI / Distributed Computing & System Integration
// Covers (Section 6A of the synopsis):
//   - MPI initialization / rank management
//   - Dataset partitioning across MPI ranks
//   - Local data distribution (MPI_Scatterv)
//   - MPI_Allreduce-based global centroid update
//   - Communication / synchronization / overhead measurement
//   - Integration point for Aarush Arora's CUDA assignment kernel
//
// This file builds and runs standalone with a CPU assignment step
// (assign_points_cpu, from kmeans_common.h) so the MPI logic — the part
// Tirth owns — can be developed, tested, and scaling-benchmarked
// independently of the CUDA kernel. See the "CUDA INTEGRATION HOOK"
// block below for exactly where Aarush's kernel plugs in and what
// contract it must satisfy.
//
// Build:
//   mpicxx -O3 -std=c++17 -I../include src/kmeans_mpi.cpp -o bin/kmeans_mpi
//   (or: make mpi   — see ../Makefile)
//
//   To build the eventual hybrid binary against Aarush's CUDA object file:
//   mpicxx -O3 -DUSE_CUDA_KERNEL -std=c++17 -I../include \
//       src/kmeans_mpi.cpp cuda_assignment_kernel.o -lcudart -o bin/kmeans_hybrid
//
// Run:
//   mpirun -np <P> ./bin/kmeans_mpi <dataset.bin> <K> [max_iters] [tol] [seed] [out_prefix]
// -----------------------------------------------------------------------
#include "kmeans_common.h"
#include <mpi.h>

// -----------------------------------------------------------------------
// CUDA INTEGRATION HOOK
// -----------------------------------------------------------------------
// Aarush's CUDA module should expose a function with this exact
// signature (compiled separately with nvcc, linked in as an extern "C"
// symbol). When USE_CUDA_KERNEL is defined at compile time, each rank's
// local assignment step below calls the GPU kernel instead of the CPU
// loop — nothing else in the MPI control flow changes, because the
// input/output contract (contiguous row-major float buffers in, per-
// cluster double sums + int counts + assignment array + SSE out) is
// identical to assign_points_cpu().
//
//   local_points    : this rank's local_N * D floats (already resident
//                      on the host after MPI_Scatterv; the CUDA function
//                      owns the H2D copy, kernel launch, and D2H copy)
//   centroids       : K * D floats, identical on every rank after Bcast
//                      / the previous iteration's Allreduce
//   out_assign      : local_N ints
//   out_local_sums  : K * D doubles, pre-zeroed by this file each iter
//   out_local_cnts  : K ints, pre-zeroed by this file each iter
//   returns         : local (partial) SSE contribution, as a double
// -----------------------------------------------------------------------
#ifdef USE_CUDA_KERNEL
extern "C" double launch_assignment_kernel(const float* local_points, int local_N, int D,
                                            const float* centroids, int K,
                                            int* out_assign,
                                            double* out_local_sums,
                                            int* out_local_cnts);
#endif

static inline double do_assignment_step(const float* local_points, int local_N, int D,
                                         const float* centroids, int K,
                                         int* out_assign,
                                         double* out_local_sums,
                                         int* out_local_cnts) {
#ifdef USE_CUDA_KERNEL
    return launch_assignment_kernel(local_points, local_N, D, centroids, K,
                                     out_assign, out_local_sums, out_local_cnts);
#else
    return assign_points_cpu(local_points, local_N, D, centroids, K,
                              out_assign, out_local_sums, out_local_cnts);
#endif
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (world_rank == 0) {
            std::fprintf(stderr,
                "Usage: mpirun -np P %s <dataset.bin> <K> [max_iters=100] [tol=1e-4] [seed=42] [out_prefix=mpi]\n",
                argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    std::string dataset_path = argv[1];
    int K         = std::atoi(argv[2]);
    int max_iters = argc > 3 ? std::atoi(argv[3]) : 100;
    double tol    = argc > 4 ? std::atof(argv[4]) : 1e-4;
    unsigned seed = argc > 5 ? (unsigned)std::atoi(argv[5]) : 42u;
    std::string out_prefix = argc > 6 ? argv[6] : "mpi";

    int rc = 0;
    try {
        // -----------------------------------------------------------
        // 1. Rank 0 reads the header and (for now) the full dataset.
        //    Only rank 0 needs the full array in memory — it is the
        //    source for MPI_Scatterv below. For datasets too large for
        //    a single node's RAM, this is the spot to switch to
        //    chunked/streamed reads per rank (MPI-IO) — flagged in the
        //    README as a scale-out extension.
        // -----------------------------------------------------------
        int N = 0, D = 0;
        Dataset full_ds; // only meaningfully populated on rank 0

        if (world_rank == 0) {
            full_ds = read_dataset_full(dataset_path);
            N = full_ds.N;
            D = full_ds.D;
            std::fprintf(stderr, "[mpi] rank0: loaded N=%d D=%d K=%d ranks=%d from %s\n",
                         N, D, K, world_size, dataset_path.c_str());
        }

        // Broadcast problem dimensions so every rank can size its buffers.
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (K > N) {
            if (world_rank == 0) std::fprintf(stderr, "[mpi] ERROR: K > N\n");
            MPI_Finalize();
            return 1;
        }

        // -----------------------------------------------------------
        // 2. Dataset partitioning: split N points as evenly as possible
        //    across ranks. Counts/displs are in POINTS; multiply by D
        //    to get the float counts/displs MPI_Scatterv needs.
        // -----------------------------------------------------------
        std::vector<int> point_counts, point_displs;
        plan_partition(N, world_size, point_counts, point_displs);

        int local_N = point_counts[world_rank];
        std::vector<float> local_points((size_t)local_N * D);

        std::vector<int> float_counts(world_size), float_displs(world_size);
        for (int r = 0; r < world_size; ++r) {
            float_counts[r] = point_counts[r] * D;
            float_displs[r] = point_displs[r] * D;
        }

        // -----------------------------------------------------------
        // 3. Local data distribution via MPI_Scatterv (uneven chunks
        //    supported — required whenever N is not divisible by P).
        // -----------------------------------------------------------
        MPI_Scatterv(world_rank == 0 ? full_ds.data.data() : nullptr,
                     float_counts.data(), float_displs.data(), MPI_FLOAT,
                     local_points.data(), local_N * D, MPI_FLOAT,
                     0, MPI_COMM_WORLD);

        std::fprintf(stderr, "[mpi] rank%d: local_N=%d (offset %d points)\n",
                     world_rank, local_N, point_displs[world_rank]);

        // -----------------------------------------------------------
        // 4. Centroid initialization: rank 0 picks K points from the
        //    full dataset, then broadcasts so every rank starts every
        //    iteration with byte-identical centroids (no drift).
        // -----------------------------------------------------------
        std::vector<float> centroids;
        if (world_rank == 0) {
            centroids = init_centroids(full_ds, K, seed);
        } else {
            centroids.resize((size_t)K * D);
        }
        MPI_Bcast(centroids.data(), K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Rank 0 no longer needs the full dataset in memory after the
        // scatter + init sampling — free it early (matters at 100M-point
        // scale where rank 0's RSS would otherwise include N*D floats
        // on top of everything else it does).
        if (world_rank == 0) { Dataset().data.swap(full_ds.data); full_ds.N = full_ds.D = 0; }

        std::vector<int> local_assign(local_N, -1);
        std::vector<double> local_sums((size_t)K * D);
        std::vector<int> local_cnts(K);
        std::vector<double> global_sums((size_t)K * D);
        std::vector<int> global_cnts(K);

        double compute_time_total = 0.0; // this rank's assignment-step time
        double comm_time_total = 0.0;    // this rank's Allreduce/Bcast time
        int iter = 0;
        double global_sse = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_run_start = MPI_Wtime();

        for (; iter < max_iters; ++iter) {
            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_cnts.begin(), local_cnts.end(), 0);

            // ---- Local compute: distance + assignment (CPU here; CUDA
            // kernel here in the hybrid build — see hook above) ----
            double t0 = MPI_Wtime();
            double local_sse = do_assignment_step(local_points.data(), local_N, D,
                                                   centroids.data(), K,
                                                   local_assign.data(),
                                                   local_sums.data(), local_cnts.data());
            double t1 = MPI_Wtime();
            compute_time_total += (t1 - t0);

            // ---- Global reduction: combine every rank's partial
            // cluster sums/counts/SSE into the global picture. Every
            // rank needs the result to build next iteration's
            // centroids, so MPI_Allreduce (not MPI_Reduce) is the right
            // collective here. ----
            MPI_Allreduce(local_sums.data(), global_sums.data(), K * D,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(local_cnts.data(), global_cnts.data(), K,
                          MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_sse, &global_sse, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double t2 = MPI_Wtime();
            comm_time_total += (t2 - t1);

            // Every rank recomputes the SAME new centroids from the now-
            // identical global sums/counts — cheap (O(K*D)) and avoids a
            // second Bcast round-trip.
            double max_shift = 0.0;
            int empty = update_centroids(global_sums.data(), global_cnts.data(),
                                          K, D, centroids, &max_shift);

            if (world_rank == 0) {
                if (empty > 0)
                    std::fprintf(stderr, "[mpi] warning: %d empty cluster(s) at iter %d\n", empty, iter);
                std::fprintf(stderr, "[mpi] iter=%d  SSE=%.6f  max_shift=%.6f  compute=%.4fs  comm=%.4fs\n",
                             iter, global_sse, max_shift, t1 - t0, t2 - t1);
            }

            if (max_shift < tol) { iter++; break; }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_run_end = MPI_Wtime();
        double total_time = t_run_end - t_run_start;

        // -----------------------------------------------------------
        // Correctness bookkeeping (Section 15): confirm the global
        // point count is preserved after reduction, and surface
        // per-rank compute-time imbalance (min/max/avg) — this is the
        // "communication overhead" and load-imbalance data objective 6
        // and 6A ask Tirth to measure.
        // -----------------------------------------------------------
        int total_assigned = 0;
        for (int k = 0; k < K; ++k) total_assigned += global_cnts[k];

        double compute_min, compute_max, compute_sum;
        double comm_min, comm_max, comm_sum;
        MPI_Reduce(&compute_time_total, &compute_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&compute_time_total, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&compute_time_total, &compute_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_time_total, &comm_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_time_total, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_time_total, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // -----------------------------------------------------------
        // 5. Gather final assignments back to rank 0 (uneven chunks via
        //    MPI_Gatherv) for output + downstream correctness validation
        //    against the serial reference.
        // -----------------------------------------------------------
        std::vector<int> full_assign;
        if (world_rank == 0) full_assign.resize(N);
        MPI_Gatherv(local_assign.data(), local_N, MPI_INT,
                    world_rank == 0 ? full_assign.data() : nullptr,
                    point_counts.data(), point_displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            if (total_assigned != N) {
                std::fprintf(stderr, "[mpi] WARNING: assigned count %d != N %d\n", total_assigned, N);
            }

            write_centroids(out_prefix + "_centroids.txt", centroids, K, D);
            write_assignments(out_prefix + "_assignments.txt", full_assign);

            FILE* mf = std::fopen((out_prefix + "_metrics.txt").c_str(), "w");
            std::fprintf(mf,
                "N %d\nD %d\nK %d\nranks %d\niterations %d\nSSE %.6f\n"
                "total_time_s %.6f\n"
                "compute_time_avg_s %.6f\ncompute_time_min_s %.6f\ncompute_time_max_s %.6f\n"
                "comm_time_avg_s %.6f\ncomm_time_min_s %.6f\ncomm_time_max_s %.6f\n"
                "comm_overhead_fraction %.6f\n",
                N, D, K, world_size, iter, global_sse, total_time,
                compute_sum / world_size, compute_min, compute_max,
                comm_sum / world_size, comm_min, comm_max,
                (comm_sum / world_size) / total_time);
            std::fclose(mf);

            std::fprintf(stderr,
                "[mpi] DONE  ranks=%d  iters=%d  SSE=%.6f  total=%.4fs  "
                "avg_compute=%.4fs  avg_comm=%.4fs  comm_overhead=%.2f%%\n",
                world_size, iter, global_sse, total_time,
                compute_sum / world_size, comm_sum / world_size,
                100.0 * (comm_sum / world_size) / total_time);
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[mpi] rank%d ERROR: %s\n", world_rank, e.what());
        rc = 1;
    }

    MPI_Finalize();
    return rc;
}
