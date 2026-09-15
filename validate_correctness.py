#!/usr/bin/env python3
"""
validate_correctness.py
--------------------------------------------------------------------
Section 15 "Correctness and Validation":
  - K-Means can converge to different local minima depending on
    execution order, so this does NOT require identical cluster
    labels. Instead it compares:
      1. SSE (objective value) within a relative tolerance
      2. centroid sets after best-matching labels via the Hungarian
         algorithm (permutation-invariant comparison)
      3. that every point received a valid assignment and the global
         point count is preserved (both runs)

Usage:
    python3 validate_correctness.py serial mpi --tol 1e-3
    (looks for serial_metrics.txt/serial_centroids.txt and
     mpi_metrics.txt/mpi_centroids.txt in the current directory, or
     pass full path prefixes)
--------------------------------------------------------------------
"""
import argparse
import sys
import numpy as np


def load_metrics(prefix):
    d = {}
    with open(f"{prefix}_metrics.txt") as f:
        for line in f:
            k, v = line.split()
            try:
                d[k] = int(v)
            except ValueError:
                d[k] = float(v)
    return d


def load_centroids(prefix):
    with open(f"{prefix}_centroids.txt") as f:
        K, D = map(int, f.readline().split())
        c = np.loadtxt(f)
        c = c.reshape(K, D)
    return c


def load_assignments(prefix):
    return np.loadtxt(f"{prefix}_assignments.txt", dtype=int)


def best_match_centroid_distance(c1, c2):
    """Permutation-invariant centroid comparison via the Hungarian
    algorithm (falls back to greedy nearest-match if scipy is absent)."""
    K = c1.shape[0]
    cost = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)
    except ImportError:
        # Greedy fallback: fine for the small K typical of this project.
        row, col, used = [], [], set()
        for i in range(K):
            order = np.argsort(cost[i])
            for j in order:
                if j not in used:
                    row.append(i); col.append(j); used.add(j)
                    break
    return cost[row, col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix_a", help="e.g. 'serial' (reference)")
    ap.add_argument("prefix_b", help="e.g. 'mpi' or 'hybrid' (candidate)")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative SSE tolerance")
    ap.add_argument("--centroid-tol", type=float, default=1e-2, help="max allowed per-centroid distance")
    args = ap.parse_args()

    ma, mb = load_metrics(args.prefix_a), load_metrics(args.prefix_b)
    ca, cb = load_centroids(args.prefix_a), load_centroids(args.prefix_b)

    ok = True

    if ma["N"] != mb["N"] or ma["D"] != mb["D"] or ma["K"] != mb["K"]:
        print(f"FAIL: shape mismatch  a=(N={ma['N']},D={ma['D']},K={ma['K']})  "
              f"b=(N={mb['N']},D={mb['D']},K={mb['K']})")
        ok = False

    rel_sse_diff = abs(ma["SSE"] - mb["SSE"]) / max(abs(ma["SSE"]), 1e-12)
    sse_pass = rel_sse_diff <= args.tol
    print(f"{'PASS' if sse_pass else 'FAIL'}: SSE  a={ma['SSE']:.6f}  b={mb['SSE']:.6f}  "
          f"rel_diff={rel_sse_diff:.6e}  (tol={args.tol})")
    ok &= sse_pass

    dists = best_match_centroid_distance(ca, cb)
    centroid_pass = bool(np.all(dists <= args.centroid_tol))
    print(f"{'PASS' if centroid_pass else 'FAIL'}: centroids  max_matched_distance={dists.max():.6f}  "
          f"(tol={args.centroid_tol})")
    ok &= centroid_pass

    try:
        aa, ab = load_assignments(args.prefix_a), load_assignments(args.prefix_b)
        valid_a = np.all((aa >= 0) & (aa < ma["K"])) and len(aa) == ma["N"]
        valid_b = np.all((ab >= 0) & (ab < mb["K"])) and len(ab) == mb["N"]
        print(f"{'PASS' if valid_a and valid_b else 'FAIL'}: every point validly assigned, "
              f"global point count preserved (a: {len(aa)}, b: {len(ab)})")
        ok &= valid_a and valid_b
    except OSError:
        print("SKIP: assignment files not found for one/both runs")

    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
