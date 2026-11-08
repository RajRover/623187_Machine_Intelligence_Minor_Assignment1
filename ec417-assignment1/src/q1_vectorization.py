"""
EC-417 Assignment 1 – Problem 1: Vectorization and Pairwise Distances
======================================================================
Computes pairwise Euclidean distance matrices using:
  (a) explicit Python loops
  (b) NumPy broadcasting (direct)
  (c) NumPy broadcasting via the algebraic expansion  ‖xi−xj‖² = ‖xi‖²+‖xj‖²−2 xi·xj

Verifies agreement with np.allclose and benchmarks the three implementations.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Implementation 1 – Nested Python loops
# ---------------------------------------------------------------------------

def pairwise_distances_loop(X: np.ndarray) -> np.ndarray:
    """Compute the N×N matrix of pairwise Euclidean distances using nested loops.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Data matrix whose rows are data points.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Distance matrix where D[i, j] = ||X[i] - X[j]||_2.
    """
    N = X.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            diff = X[i] - X[j]
            D[i, j] = np.sqrt(np.dot(diff, diff))
    return D


# ---------------------------------------------------------------------------
# Implementation 2 – NumPy broadcasting (direct)
# ---------------------------------------------------------------------------

def pairwise_distances_broadcast(X: np.ndarray) -> np.ndarray:
    """Compute the N×N pairwise Euclidean distance matrix using NumPy broadcasting.

    Intermediate array shape: (N, N, d).  For N=2000, d=20 in float64 this is
    2000×2000×20×8 bytes ≈ 640 MB peak memory.  Memory scales as O(N²·d).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Data matrix whose rows are data points.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Distance matrix where D[i, j] = ||X[i] - X[j]||_2.
    """
    diff = X[:, None, :] - X[None, :, :]   # shape (N, N, d)
    D = np.sqrt((diff ** 2).sum(axis=-1))   # shape (N, N)
    return D


# ---------------------------------------------------------------------------
# Implementation 3 – Algebraic expansion (memory-efficient vectorized)
# ---------------------------------------------------------------------------

def pairwise_distances_expansion(X: np.ndarray) -> np.ndarray:
    """Compute the N×N pairwise Euclidean distance matrix via the algebraic expansion.

    Uses  ‖xi−xj‖² = ‖xi‖² + ‖xj‖² − 2 xi·xj  to avoid the large (N,N,d)
    intermediate array.  Peak memory is dominated by the (N,N) inner-product
    matrix → O(N²) instead of O(N²·d).

    Floating-point roundoff may produce small negative values (~−10⁻¹⁴) under
    the square-root; these are clipped to 0 with np.maximum before calling sqrt.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Data matrix whose rows are data points.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Distance matrix where D[i, j] = ||X[i] - X[j]||_2.
    """
    sq_norms = (X ** 2).sum(axis=1)                    # shape (N,)
    # Gram matrix via BLAS matrix-multiply – very fast
    gram = X @ X.T                                      # shape (N, N)
    # sq_norms[:, None] + sq_norms[None, :] broadcasts over both axes
    D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    # Clip negative values caused by floating-point roundoff before sqrt
    D = np.sqrt(np.maximum(0.0, D_sq))
    return D


# ---------------------------------------------------------------------------
# Verification and benchmarking
# ---------------------------------------------------------------------------

def verify_and_benchmark(N: int = 2000, d: int = 20,
                          seed: int = 42) -> dict:
    """Verify all three implementations agree and benchmark their runtimes.

    Parameters
    ----------
    N : int
        Number of data points.
    d : int
        Dimensionality.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Keys: 'loop_time', 'broadcast_time', 'expansion_time',
              'loop_vs_broadcast_ok', 'loop_vs_expansion_ok',
              'speedup_broadcast', 'speedup_expansion'.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))

    # --- small subset for the loop (it is O(N²·d) in pure Python) ---
    N_small = 200  # timing on 200 points, then extrapolate is impractical;
                   # we time on full N for broadcast/expansion, small N for loop
    X_small = X[:N_small]

    print(f"Verifying correctness on N={N_small}, d={d} ...")
    D_loop = pairwise_distances_loop(X_small)
    D_broad = pairwise_distances_broadcast(X_small)
    D_exp = pairwise_distances_expansion(X_small)

    # The direct broadcasting method accumulates error at the same order as
    # np.sqrt of machine epsilon (~1e-12), so agreement at 1e-10 is expected.
    ok_1 = np.allclose(D_loop, D_broad, atol=1e-10)
    # The algebraic expansion uses matrix multiply + subtraction; catastrophic
    # cancellation when xi~xj leads to larger roundoff (~1e-7), verified at
    # a numerically appropriate tolerance of 1e-6.
    ok_2_strict = np.allclose(D_loop, D_exp, atol=1e-10)
    ok_2_loose  = np.allclose(D_loop, D_exp, atol=1e-6)
    max_diff_exp = float(np.abs(D_loop - D_exp).max())
    print(f"  loop ~= broadcast  (atol=1e-10): {ok_1}")
    print(f"  loop ~= expansion  (atol=1e-10): {ok_2_strict}  "
          f"[max diff = {max_diff_exp:.2e}; expected -- see note below]")
    print(f"  loop ~= expansion  (atol=1e-6) : {ok_2_loose}")
    print(f"  Note: the expansion method trades numerical precision (~1e-7) for")
    print(f"  lower memory (O(N^2) vs O(N^2*d)); clip before sqrt prevents NaN.")

    # --- Timing loop on N_small (extrapolate note) ---
    print(f"\nTiming loop on N={N_small}, d={d} …")
    t0 = time.perf_counter()
    pairwise_distances_loop(X_small)
    loop_time_small = time.perf_counter() - t0
    # O(N²) scaling → estimated time for N=2000
    loop_time_estimated = loop_time_small * (N / N_small) ** 2
    print(f"  Actual time for N={N_small}: {loop_time_small:.3f} s")
    print(f"  Estimated time for N={N}: {loop_time_estimated:.1f} s (O(N²) extrapolation)")

    # --- Timing broadcasting on full N ---
    print(f"\nTiming broadcasting on N={N}, d={d} …")
    t0 = time.perf_counter()
    pairwise_distances_broadcast(X)
    broadcast_time = time.perf_counter() - t0
    print(f"  Broadcasting time: {broadcast_time:.4f} s")

    # --- Timing expansion on full N ---
    print(f"\nTiming expansion on N={N}, d={d} …")
    t0 = time.perf_counter()
    pairwise_distances_expansion(X)
    expansion_time = time.perf_counter() - t0
    print(f"  Expansion time: {expansion_time:.4f} s")

    speedup_broadcast = loop_time_estimated / broadcast_time
    speedup_expansion = loop_time_estimated / expansion_time

    # --- Memory estimates ---
    intermediate_broadcast_MB = N * N * d * 8 / 1e6
    intermediate_expansion_MB = N * N * 8 / 1e6
    print(f"\nPeak intermediate memory:")
    print(f"  Broadcasting (N,N,d) array: {intermediate_broadcast_MB:.1f} MB")
    print(f"  Expansion    (N,N)   array: {intermediate_expansion_MB:.1f} MB")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print(f"{'Method':<25} {'Time (s)':<15} {'Speedup vs loop'}")
    print("=" * 60)
    print(f"{'Loop (extrapolated)':<25} {loop_time_estimated:<15.2f} {'1×'}")
    print(f"{'Broadcasting (direct)':<25} {broadcast_time:<15.4f} {speedup_broadcast:.0f}×")
    print(f"{'Expansion (matmul)':<25} {expansion_time:<15.4f} {speedup_expansion:.0f}×")
    print("=" * 60)
    print("\nThe speedup is large because NumPy delegates the inner loop to")
    print("optimised C/BLAS routines that leverage SIMD and cache-friendly")
    print("memory access, whereas pure Python loops carry interpreter overhead")
    print("per element (~100× slower).\n")
    print("Recommendation: Use the algebraic-expansion method for large ML")
    print("datasets.  It is as fast as (or faster than) direct broadcasting,")
    print("uses d× less peak memory (O(N²) vs O(N²d)), and is numerically")
    print("stable after clipping.  Direct broadcasting is simpler to read but")
    print("can exhaust RAM for large N or d.")

    return {
        "loop_time_small": loop_time_small,
        "loop_time_estimated": loop_time_estimated,
        "broadcast_time": broadcast_time,
        "expansion_time": expansion_time,
        "loop_vs_broadcast_ok": ok_1,
        "loop_vs_expansion_ok_1e10": ok_2_strict,
        "loop_vs_expansion_ok_1e6": ok_2_loose,
        "expansion_max_diff": max_diff_exp,
        "speedup_broadcast": speedup_broadcast,
        "speedup_expansion": speedup_expansion,
        "peak_mem_broadcast_MB": intermediate_broadcast_MB,
        "peak_mem_expansion_MB": intermediate_expansion_MB,
    }


if __name__ == "__main__":
    results = verify_and_benchmark(N=2000, d=20, seed=42)
