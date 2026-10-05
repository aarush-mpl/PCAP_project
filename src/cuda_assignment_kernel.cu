// src/cuda_assignment_kernel.cu
// -----------------------------------------------------------------------
// CUDA GPU Assignment Kernel for Scalable Parallel K-Means (PS-5).
//
// Implements point-to-centroid assignment acceleration on NVIDIA GPUs.
// Exposes the exact extern "C" contract expected by src/kmeans_mpi.cpp
// when compiled with -DUSE_CUDA_KERNEL.
//
// Includes runtime GPU availability detection and automatic CPU fallback
// if CUDA device or driver is unavailable (e.g. in virtualized WSL).
//
// Owner: Dispo / Tirth Integration (GPU-side assignment workstream)
// -----------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "kmeans_common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>

// Atomic double addition fallback for older CUDA architectures (< 6.0)
#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// -----------------------------------------------------------------------
// CUDA Kernel: Each thread computes assignment for one local point.
// -----------------------------------------------------------------------
__global__ void kmeans_assign_kernel(const float* __restrict__ d_points,
                                      int local_N, int D,
                                      const float* __restrict__ d_centroids,
                                      int K,
                                      int* __restrict__ d_assign,
                                      double* __restrict__ d_sums,
                                      int* __restrict__ d_cnts,
                                      double* __restrict__ d_sse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_N) return;

    const float* p = d_points + (size_t)idx * D;
    int best_k = 0;
    double best_dist = 1e300;

    for (int k = 0; k < K; ++k) {
        const float* c = d_centroids + (size_t)k * D;
        double dist = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = (double)p[d] - (double)c[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }

    d_assign[idx] = best_k;
    atomicAdd(&d_cnts[best_k], 1);
    atomicAddDouble(d_sse, best_dist);

    double* sum_row = d_sums + (size_t)best_k * D;
    for (int d = 0; d < D; ++d) {
        atomicAddDouble(&sum_row[d], (double)p[d]);
    }
}

extern "C" double launch_assignment_kernel(const float* local_points, int local_N, int D,
                                            const float* centroids, int K,
                                            int* out_assign,
                                            double* out_local_sums,
                                            int* out_local_cnts) {
    if (local_N <= 0) return 0.0;

    // Check if CUDA GPU device is available at runtime
    int device_count = 0;
    cudaError_t err_dev = cudaGetDeviceCount(&device_count);
    if (err_dev != cudaSuccess || device_count == 0) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "[cuda] Warning: No active CUDA GPU device detected (error: %s). Falling back to CPU.\n",
                         cudaGetErrorString(err_dev));
            warned = true;
        }
        return assign_points_cpu(local_points, local_N, D, centroids, K,
                                 out_assign, out_local_sums, out_local_cnts);
    }

    float *d_points = nullptr, *d_centroids = nullptr;
    int *d_assign = nullptr, *d_cnts = nullptr;
    double *d_sums = nullptr, *d_sse = nullptr;

    size_t points_bytes = (size_t)local_N * D * sizeof(float);
    size_t centroids_bytes = (size_t)K * D * sizeof(float);
    size_t assign_bytes = (size_t)local_N * sizeof(int);
    size_t sums_bytes = (size_t)K * D * sizeof(double);
    size_t cnts_bytes = (size_t)K * sizeof(int);

    cudaError_t err1 = cudaMalloc((void**)&d_points, points_bytes);
    cudaError_t err2 = cudaMalloc((void**)&d_centroids, centroids_bytes);
    cudaError_t err3 = cudaMalloc((void**)&d_assign, assign_bytes);
    cudaError_t err4 = cudaMalloc((void**)&d_sums, sums_bytes);
    cudaError_t err5 = cudaMalloc((void**)&d_cnts, cnts_bytes);
    cudaError_t err6 = cudaMalloc((void**)&d_sse, sizeof(double));

    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess ||
        err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
        std::fprintf(stderr, "[cuda] Error: GPU memory allocation failed. Falling back to CPU.\n");
        if (d_points) cudaFree(d_points);
        if (d_centroids) cudaFree(d_centroids);
        if (d_assign) cudaFree(d_assign);
        if (d_sums) cudaFree(d_sums);
        if (d_cnts) cudaFree(d_cnts);
        if (d_sse) cudaFree(d_sse);
        return assign_points_cpu(local_points, local_N, D, centroids, K,
                                 out_assign, out_local_sums, out_local_cnts);
    }

    cudaMemcpy(d_points, local_points, points_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, centroids_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_sums, 0, sums_bytes);
    cudaMemset(d_cnts, 0, cnts_bytes);
    cudaMemset(d_sse, 0, sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (local_N + threadsPerBlock - 1) / threadsPerBlock;

    kmeans_assign_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_points, local_N, D, d_centroids, K, d_assign, d_sums, d_cnts, d_sse
    );

    cudaDeviceSynchronize();

    double local_sse = 0.0;
    cudaMemcpy(out_assign, d_assign, assign_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_local_sums, d_sums, sums_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_local_cnts, d_cnts, cnts_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&local_sse, d_sse, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_assign);
    cudaFree(d_sums);
    cudaFree(d_cnts);
    cudaFree(d_sse);

    return local_sse;
}

#else

// -----------------------------------------------------------------------
// Host Fallback: Allows compiling/testing without NVCC/CUDA SDK
// -----------------------------------------------------------------------
extern "C" double launch_assignment_kernel(const float* local_points, int local_N, int D,
                                            const float* centroids, int K,
                                            int* out_assign,
                                            double* out_local_sums,
                                            int* out_local_cnts) {
    return assign_points_cpu(local_points, local_N, D, centroids, K,
                             out_assign, out_local_sums, out_local_cnts);
}

#endif
