#!/usr/bin/env python3
"""
prepare_real_data.py
--------------------------------------------------------------------
Section 10 "Dataset and Benchmark Plan" real-world validation data:
downloads and preprocesses MNIST, KDD Cup 1999, and US Census (1990)
into the project's shared binary format (see include/kmeans_common.h)
so the SAME serial / MPI / hybrid binaries run on real data as on the
synthetic benchmark data.

Sources used (all standard, citable, maintained loaders -- not scraped
mirrors):
  - MNIST        : sklearn.datasets.fetch_openml('mnist_784', ...)
                    (pulls from openml.org, the standard MNIST-as-CSV
                    mirror used throughout the ML community)
  - KDD Cup 1999  : sklearn.datasets.fetch_kddcup99(...)
                    (scikit-learn's own built-in loader for this exact
                    benchmark, pulling from the UCI KDD Archive)
  - US Census 1990: ucimlrepo.fetch_ucirepo(id=116)
                    (official UCI ML Repository Python client;
                    https://archive.ics.uci.edu/dataset/116)

Each loader downloads to a local cache on first use (~/scikit_learn_data
for the sklearn loaders) so repeat runs / repeat experiments don't
re-download.

Usage:
    python3 scripts/prepare_real_data.py --dataset mnist \
        --out data/real_mnist.bin

    python3 scripts/prepare_real_data.py --dataset kddcup99 --percent10 \
        --out data/real_kddcup99_10pct.bin

    python3 scripts/prepare_real_data.py --dataset uscensus1990 \
        --max-n 2000000 --out data/real_uscensus1990.bin

Requires (install once, on a machine with internet access):
    pip install scikit-learn pandas ucimlrepo
--------------------------------------------------------------------
"""
import argparse
import struct
import sys
import time

import numpy as np

MAGIC = 0x4B4D4E53


# -----------------------------------------------------------------------
# Loaders — each returns (X: float32 ndarray [N, D], feature_names: list[str],
# info: dict of source/citation metadata for the report).
# -----------------------------------------------------------------------

def load_mnist():
    from sklearn.datasets import fetch_openml
    print("Fetching MNIST (mnist_784) from OpenML — first run downloads "
          "~55MB and caches it locally...")
    bunch = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = bunch.data.astype(np.float32)
    # Standard preprocessing for pixel data: scale 0-255 -> 0-1.
    X /= 255.0
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    info = {
        "source": "OpenML mnist_784 (via scikit-learn fetch_openml)",
        "citation": "LeCun, Y., Cortes, C., Burges, C.J.C. THE MNIST DATABASE. "
                    "http://yann.lecun.com/exdb/mnist/",
        "preprocessing": "pixel intensities scaled from [0,255] to [0,1]",
    }
    return X, feature_names, info


def load_kddcup99(percent10: bool):
    from sklearn.datasets import fetch_kddcup99
    print(f"Fetching KDD Cup 1999 ({'10%' if percent10 else 'full'} subset) "
          f"via scikit-learn fetch_kddcup99 — first run downloads and caches "
          f"locally...")
    bunch = fetch_kddcup99(percent10=percent10, as_frame=True)
    df = bunch.frame.copy()
    df = df.drop(columns=['labels'])  # this is the intrusion-type label, not a feature

    # 3 categorical columns need one-hot encoding; the rest are already
    # numeric (bytes come back as Python bytes objects from this loader
    # so cast explicitly).
    cat_cols = ['protocol_type', 'service', 'flag']
    for c in cat_cols:
        df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    return _kddcup99_to_matrix(df, cat_cols)


def _kddcup99_to_matrix(df, cat_cols):
    import pandas as pd
    df = pd.get_dummies(df, columns=cat_cols)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.float32)
    X = df.to_numpy(dtype=np.float32)
    feature_names = list(df.columns)
    info = {
        "source": "UCI KDD Archive, KDD Cup 1999 Data (via scikit-learn fetch_kddcup99)",
        "citation": "Stolfo, S., Fan, W., Lee, W., Prodromidis, A., & Chan, P. (1999). "
                    "KDD Cup 1999 Data [Dataset]. UCI Machine Learning Repository. "
                    "https://doi.org/10.24432/C51C7N",
        "preprocessing": f"one-hot encoded categorical columns {cat_cols}; "
                          f"'labels' (attack type) column dropped -- unsupervised task",
    }
    return X, feature_names, info


def load_uscensus1990(max_n: int | None):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        sys.exit("ucimlrepo is required for --dataset uscensus1990.\n"
                  "Install it with:  pip install ucimlrepo")
    print("Fetching US Census Data (1990), UCI dataset id=116 — first run "
          "downloads and caches locally...")
    ds = fetch_ucirepo(id=116)
    df = ds.data.features.copy()
    # 'caseid' is a record identifier, not a feature -- drop if present.
    if 'caseid' in df.columns:
        df = df.drop(columns=['caseid'])
    if max_n is not None and len(df) > max_n:
        df = df.sample(n=max_n, random_state=42)

    X = df.to_numpy(dtype=np.float32)
    feature_names = list(df.columns)
    info = {
        "source": "UCI Machine Learning Repository, US Census Data (1990), id=116",
        "citation": "Meek, C., Thiesson, B., & Heckerman, D. (2001). "
                    "US Census Data (1990) [Dataset]. UCI Machine Learning Repository. "
                    "https://doi.org/10.24432/C5VP42",
        "preprocessing": "'caseid' identifier column dropped; all remaining fields "
                          "are already integer-coded (categorical/ordinal) by the "
                          "source -- used here directly as numeric features for "
                          "scalability benchmarking. NOTE: because these codes are "
                          "categorical rather than continuous measurements, treat "
                          "resulting clusters as a scale/throughput benchmark "
                          "primarily, not a semantically validated segmentation "
                          "(same caveat applies to any Euclidean K-Means run on "
                          "one-hot / ordinal-coded categorical data).",
    }
    return X, feature_names, info


# -----------------------------------------------------------------------
# Shared pipeline: standardize, write .bin (same format kmeans_common.h
# reads), write a companion report for the project documentation.
# -----------------------------------------------------------------------

def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # guard against constant columns (avoid div-by-zero)
    return ((X - mean) / std).astype(np.float32)


def write_bin(path: str, X: np.ndarray):
    N, D = X.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<i", MAGIC))
        f.write(struct.pack("<i", N))
        f.write(struct.pack("<i", D))
        X.astype(np.float32, copy=False).tofile(f)


def write_report(path: str, dataset_name: str, X: np.ndarray, feature_names,
                  info: dict, standardized: bool, elapsed_s: float):
    with open(path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Shape: N={X.shape[0]}  D={X.shape[1]}\n")
        f.write(f"Standardized (zero mean / unit variance per feature): {standardized}\n")
        f.write(f"Preparation time: {elapsed_s:.1f}s\n\n")
        f.write(f"Source: {info['source']}\n")
        f.write(f"Citation: {info['citation']}\n")
        f.write(f"Preprocessing: {info['preprocessing']}\n\n")
        f.write(f"Feature count: {len(feature_names)}\n")
        if len(feature_names) <= 60:
            f.write("Features: " + ", ".join(feature_names) + "\n")
        else:
            f.write("Features (first 30 of %d): %s ...\n" %
                    (len(feature_names), ", ".join(feature_names[:30])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                     choices=["mnist", "kddcup99", "uscensus1990"])
    ap.add_argument("--out", required=True, help="output .bin path")
    ap.add_argument("--percent10", action="store_true",
                     help="[kddcup99 only] use the standard 10%% subset "
                          "(~494K rows) instead of the full ~4.9M-row dataset")
    ap.add_argument("--max-n", type=int, default=None,
                     help="optionally cap the number of points (random "
                          "subsample) -- useful for quick local testing "
                          "before a full-scale run")
    ap.add_argument("--no-standardize", action="store_true",
                     help="skip zero-mean/unit-variance standardization "
                          "(K-Means with squared-Euclidean distance is "
                          "scale-sensitive, so standardizing is recommended "
                          "unless you have a specific reason not to)")
    args = ap.parse_args()

    t0 = time.time()

    if args.dataset == "mnist":
        X, feature_names, info = load_mnist()
    elif args.dataset == "kddcup99":
        X, feature_names, info = load_kddcup99(percent10=args.percent10)
    else:
        X, feature_names, info = load_uscensus1990(max_n=args.max_n)

    # Generic subsampling path (mnist / kddcup99 didn't apply --max-n above).
    if args.max_n is not None and args.dataset != "uscensus1990" and X.shape[0] > args.max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=args.max_n, replace=False)
        X = X[idx]

    print(f"Loaded {args.dataset}: N={X.shape[0]:,}  D={X.shape[1]}")

    standardized = not args.no_standardize
    if standardized:
        print("Standardizing features (zero mean, unit variance)...")
        X = standardize(X)

    write_bin(args.out, X)
    report_path = args.out.rsplit(".", 1)[0] + "_report.txt"
    write_report(report_path, args.dataset, X, feature_names, info,
                 standardized, time.time() - t0)

    print(f"Wrote {args.out}  ({X.shape[0]:,} x {X.shape[1]}, "
          f"{X.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
