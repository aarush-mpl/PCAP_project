// kmeans_serial.cpp
// -----------------------------------------------------------------------
// Serial CPU K-Means reference implementation (Objective 1 / Section 13
// "Serial CPU K-Means: single-thread reference implementation").
//
// This is the correctness + performance baseline every other version
// (OpenMP, CUDA, hybrid MPI+CUDA) is validated and compared against.
//
// Build:  make serial        (see ../Makefile)
// Run:    ./bin/kmeans_serial <dataset.bin> <K> [max_iters] [tol] [seed] [out_prefix]
// -----------------------------------------------------------------------
#include "kmeans_common.h"
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <dataset.bin> <K> [max_iters=100] [tol=1e-4] [seed=42] [out_prefix=serial]\n",
            argv[0]);
        return 1;
    }

    std::string dataset_path = argv[1];
    int K            = std::atoi(argv[2]);
    int max_iters    = argc > 3 ? std::atoi(argv[3]) : 100;
    double tol       = argc > 4 ? std::atof(argv[4]) : 1e-4;
    unsigned seed    = argc > 5 ? (unsigned)std::atoi(argv[5]) : 42u;
    std::string out_prefix = argc > 6 ? argv[6] : "serial";

    try {
        auto t_load_start = Clock::now();
        Dataset ds = read_dataset_full(dataset_path);
        auto t_load_end = Clock::now();

        std::fprintf(stderr, "[serial] Loaded N=%d D=%d K=%d from %s\n",
                     ds.N, ds.D, K, dataset_path.c_str());

        std::vector<float> centroids = init_centroids(ds, K, seed);
        std::vector<int> assign(ds.N, -1);

        std::vector<double> sums((size_t)K * ds.D);
        std::vector<int> cnts(K);

        int iter = 0;
        double sse = 0.0;
        auto t_compute_start = Clock::now();
        for (; iter < max_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(cnts.begin(), cnts.end(), 0);

            sse = assign_points_cpu(ds.data.data(), ds.N, ds.D,
                                     centroids.data(), K,
                                     assign.data(), sums.data(), cnts.data());

            double max_shift = 0.0;
            int empty = update_centroids(sums.data(), cnts.data(), K, ds.D, centroids, &max_shift);
            if (empty > 0) {
                std::fprintf(stderr, "[serial] warning: %d empty cluster(s) at iter %d\n", empty, iter);
            }

            std::fprintf(stderr, "[serial] iter=%d  SSE=%.6f  max_centroid_shift=%.6f\n",
                         iter, sse, max_shift);

            if (max_shift < tol) { iter++; break; }
        }
        auto t_compute_end = Clock::now();

        double load_s   = std::chrono::duration<double>(t_load_end - t_load_start).count();
        double compute_s = std::chrono::duration<double>(t_compute_end - t_compute_start).count();

        // Validation per Section 15: every point has a valid assignment,
        // global point count preserved.
        int total_assigned = 0;
        for (int k = 0; k < K; ++k) total_assigned += cnts[k];
        if (total_assigned != ds.N) {
            std::fprintf(stderr, "[serial] WARNING: assigned count %d != N %d\n", total_assigned, ds.N);
        }

        write_centroids(out_prefix + "_centroids.txt", centroids, K, ds.D);
        write_assignments(out_prefix + "_assignments.txt", assign);

        FILE* mf = std::fopen((out_prefix + "_metrics.txt").c_str(), "w");
        std::fprintf(mf, "N %d\nD %d\nK %d\niterations %d\nSSE %.6f\n"
                          "load_time_s %.6f\ncompute_time_s %.6f\ntotal_time_s %.6f\n",
                     ds.N, ds.D, K, iter, sse, load_s, compute_s, load_s + compute_s);
        std::fclose(mf);

        std::fprintf(stderr,
            "[serial] DONE  iters=%d  SSE=%.6f  load=%.4fs  compute=%.4fs  total=%.4fs\n",
            iter, sse, load_s, compute_s, load_s + compute_s);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[serial] ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
