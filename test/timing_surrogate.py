"""Time surrogate forward evaluation (GP predict + POD back to physical space)."""

from __future__ import annotations

import os
import pickle
import time

import numpy as np

from workflows.schema import BETA_RANGE, T_FINAL

# Default: trained artifacts (override if your run lives elsewhere)
SURROGATE_PATH = "results/outputs/uncertainty_model/surrogate.pkl"
POD_PATH = "results/outputs/uncertainty_model/pod.pkl"

N_BETAS = 50
WARMUP = 3


def load_models(surrogate_file: str, pod_file: str):
    if os.path.exists(surrogate_file) and os.path.exists(pod_file):
        with open(surrogate_file, "rb") as f:
            surrogate = pickle.load(f)
        with open(pod_file, "rb") as f:
            pod = pickle.load(f)
        return surrogate, pod
    raise FileNotFoundError(
        f"Surrogate or POD file not found at {surrogate_file} or {pod_file}"
    )


def surrogate_forward(surrogate, pod, X: np.ndarray) -> np.ndarray:
    """One pipeline-style forward: coefficients then inverse POD."""
    coeffs = surrogate.predict(X)
    return pod.svd.inverse_transform(coeffs)


if __name__ == "__main__":
    surrogate, pod = load_models(SURROGATE_PATH, POD_PATH)

    betas = np.linspace(BETA_RANGE[0], BETA_RANGE[1], N_BETAS)
    X_query = np.column_stack([betas, np.full(N_BETAS, T_FINAL)])


    #Warm-up (JIT, caches, first sklearn allocations)
    for _ in range(WARMUP):
        surrogate_forward(surrogate, pod, X_query[:1])

    per_query_s: list[float] = []
    for i in range(N_BETAS):
        x = X_query[i : i + 1]
        t0 = time.perf_counter()
        result = surrogate_forward(surrogate, pod, x)
        if i == 5:
            print(x[0, 0])
            print(result)
        per_query_s.append(time.perf_counter() - t0)

    arr = np.asarray(per_query_s)
    total = float(arr.sum())

    print(f"Queries: {N_BETAS} (one (beta, t) each, t = {T_FINAL})")
    print(f"Total surrogate time: {total:.6f} s")
    print(f"Mean per query: {arr.mean() * 1e3:.4f} ms")
    print(f"Median per query: {np.median(arr) * 1e3:.4f} ms")
    print(f"Std per query: {arr.std() * 1e3:.4f} ms")
