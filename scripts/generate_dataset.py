#!/usr/bin/env python3
"""
generate_dataset.py
--------------------------------------------------------------------
Synthetic clustered dataset generator for the K-Means scaling
benchmarks (Section 10 "Dataset and Benchmark Plan").

Writes the exact binary format read by kmeans_common.h:
    int32 magic (0x4B4D4E53 = "KMNS")
    int32 N
    int32 D
    float32 data[N*D]   (row-major)

Usage:
    python3 generate_dataset.py --n 1000000 --d 16 --k 20 --out data/synth_1M_d16_k20.bin
    python3 generate_dataset.py --n 100000000 --d 32 --k 50 --out data/synth_100M_d32_k50.bin --seed 7

Owner: Tirth Patel (experiment scripts / reproducibility configs)
--------------------------------------------------------------------
"""
import argparse
import struct
import sys
import numpy as np

MAGIC = 0x4B4D4E53


def generate(n_points: int, dim: int, n_clusters: int, spread: float, seed: int,
             chunk_size: int = 2_000_000):
    """Yield the dataset in chunks so multi-hundred-million-point runs
    don't require holding the whole array in RAM twice."""
    rng = np.random.default_rng(seed)

    # Ground-truth cluster centers, spread out in [-scale, scale]^D so
    # clusters are (mostly) well separated -- gives SSE/ARI something
    # meaningful to validate against.
    scale = spread * np.sqrt(n_clusters)
    true_centers = rng.uniform(-scale, scale, size=(n_clusters, dim)).astype(np.float32)

    remaining = n_points
    while remaining > 0:
        take = min(chunk_size, remaining)
        cluster_ids = rng.integers(0, n_clusters, size=take)
        noise = rng.normal(0.0, spread, size=(take, dim)).astype(np.float32)
        chunk = true_centers[cluster_ids] + noise
        yield chunk.astype(np.float32)
        remaining -= take


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic clustered dataset for K-Means benchmarks.")
    ap.add_argument("--n", type=int, required=True, help="number of points")
    ap.add_argument("--d", type=int, required=True, help="dimensionality")
    ap.add_argument("--k", type=int, required=True, help="number of ground-truth clusters")
    ap.add_argument("--spread", type=float, default=1.0, help="within-cluster std dev (default 1.0)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--out", type=str, required=True, help="output .bin path")
    args = ap.parse_args()

    if args.k > args.n:
        sys.exit("--k cannot exceed --n")

    with open(args.out, "wb") as f:
        f.write(struct.pack("<i", MAGIC))
        f.write(struct.pack("<i", args.n))
        f.write(struct.pack("<i", args.d))
        written = 0
        for chunk in generate(args.n, args.d, args.k, args.spread, args.seed):
            chunk.tofile(f)
            written += chunk.shape[0]
            print(f"\r  wrote {written:,}/{args.n:,} points", end="", flush=True)
        print()

    print(f"Dataset written to {args.out}  (N={args.n}, D={args.d}, "
          f"K_true={args.k}, size={args.n * args.d * 4 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
