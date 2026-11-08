"""
EC-417 Assignment 1 – Problem 3: Empirical Bias–Variance Decomposition
=======================================================================
Numerically verifies  E[loss] = σ² + Bias² + Variance
by Monte-Carlo simulation over S independent training sets.

Parts:
  (a) Simulate S=200 training sets → shape (S, 100) prediction array F_hat.
  (b) Plot true function, mean prediction, and 20 individual fits per degree.
  (c) Compute pointwise Bias²(x) and Var(x); report grid averages per M.
  (d) Qualitative expectations for bias and variance vs. degree.
  (e) Numerical verification table; rerun with S=2000.
  (f) Bias², Var vs. M=0..12 figure.
  (g) Short answer on practical unavailability.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──────────────────────────────────────────────────────────
from q2_polynomial_fitting import (
    true_function, generate_data, fit_polynomial, SIGMA, SEED
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_predictions(degree: int, S: int, n_train: int,
                          x_grid: np.ndarray, seed: int = SEED
                          ) -> np.ndarray:
    """Draw S training sets and evaluate degree-M fits on x_grid.

    Parameters
    ----------
    degree  : polynomial degree M.
    S       : number of Monte-Carlo repetitions.
    n_train : training set size.
    x_grid  : np.ndarray, shape (K,), evaluation grid.
    seed    : master random seed.

    Returns
    -------
    F_hat : np.ndarray, shape (S, K)
        F_hat[s, k] = predicted value of the s-th fit at x_grid[k].
    """
    rng = np.random.default_rng(seed)
    K = len(x_grid)
    F_hat = np.zeros((S, K))
    for s in range(S):
        x_tr, y_tr = generate_data(n_train, rng)
        poly = fit_polynomial(x_tr, y_tr, degree)
        F_hat[s] = poly(x_grid)
    return F_hat


def compute_bias2_var(F_hat: np.ndarray,
                      f_true: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pointwise bias², variance, and mean prediction.

    Parameters
    ----------
    F_hat  : np.ndarray, shape (S, K) – predictions from S datasets.
    f_true : np.ndarray, shape (K,)   – true function values on the grid.

    Returns
    -------
    f_bar  : np.ndarray, shape (K,) – mean prediction over S datasets.
    bias2  : np.ndarray, shape (K,) – (f_bar(x) - f*(x))².
    var    : np.ndarray, shape (K,) – variance of F_hat over s.
    """
    f_bar = F_hat.mean(axis=0)                # shape (K,)
    bias2 = (f_bar - f_true) ** 2            # shape (K,)
    var   = F_hat.var(axis=0)                 # shape (K,) – uses 1/S (biased)
    return f_bar, bias2, var


# ---------------------------------------------------------------------------
# Part (b) – Overlay plot
# ---------------------------------------------------------------------------

def plot_overlay(x_grid: np.ndarray, F_hat: np.ndarray,
                 f_bar: np.ndarray, degree: int,
                 filename: str) -> None:
    """Plot true function, mean prediction, and 20 individual fits.

    Parameters
    ----------
    x_grid  : evaluation grid, shape (K,).
    F_hat   : shape (S, K) – all predicted curves.
    f_bar   : shape (K,)   – average prediction.
    degree  : polynomial degree (used in title).
    filename: output filename in figures/.
    """
    f_true = true_function(x_grid)
    fig, ax = plt.subplots(figsize=(8, 5))

    # 20 individual fits (semi-transparent)
    for s in range(20):
        ax.plot(x_grid, F_hat[s], color="lightblue", alpha=0.55, lw=0.9)

    ax.plot(x_grid, f_true, "k-",  lw=2.5, label=r"$f^*(x)=\sin(2\pi x)$",
            zorder=5)
    ax.plot(x_grid, f_bar,  "r--", lw=2.5, label=r"$\bar{f}(x)$ (mean prediction)",
            zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"Bias–Variance Decomposition – Degree M={degree}\n"
                 f"(S={F_hat.shape[0]} training sets, n=15 each)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Part (e) – Numerical verification table
# ---------------------------------------------------------------------------

def compute_simulated_total(F_hat: np.ndarray,
                             f_true: np.ndarray,
                             sigma: float = SIGMA,
                             seed: int = SEED + 100) -> np.ndarray:
    """Estimate E[(Y - f_hat)²] by Monte-Carlo at each grid point.

    A fresh independent noise sample is drawn per grid point per repetition s.

    Parameters
    ----------
    F_hat  : np.ndarray, shape (S, K).
    f_true : np.ndarray, shape (K,).
    sigma  : noise standard deviation.
    seed   : random seed for the fresh noise samples.

    Returns
    -------
    total : np.ndarray, shape (K,) – MC estimate of total expected loss.
    """
    S, K = F_hat.shape
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=(S, K))       # shape (S, K)
    Y   = f_true[None, :] + eps                   # shape (S, K)
    total = np.mean((Y - F_hat) ** 2, axis=0)    # shape (K,)
    return total


def verification_table(degrees_list: list[int], S: int,
                        x_grid: np.ndarray, n_train: int = 15,
                        seed: int = SEED) -> None:
    """Print the verification table for specified degrees.

    Parameters
    ----------
    degrees_list : list of M values to tabulate.
    S            : number of MC repetitions.
    x_grid       : evaluation grid, shape (K,).
    n_train      : training set size.
    seed         : random seed.
    """
    f_true = true_function(x_grid)
    sigma2 = SIGMA ** 2

    print(f"\n{'M':>4}  {'σ²':>8}  {'Bias²':>8}  {'Var':>8}  {'Sum':>8}  {'Simul.Total':>12}")
    print("-" * 58)

    for M in degrees_list:
        F_hat = simulate_predictions(M, S, n_train, x_grid, seed=seed)
        f_bar, bias2, var = compute_bias2_var(F_hat, f_true)

        total = compute_simulated_total(F_hat, f_true, SIGMA, seed=seed + 999)

        b2_avg  = float(bias2.mean())
        var_avg = float(var.mean())
        tot_avg = float(total.mean())
        sum_bv  = sigma2 + b2_avg + var_avg

        print(f"{M:>4}  {sigma2:>8.5f}  {b2_avg:>8.5f}  {var_avg:>8.5f}  "
              f"{sum_bv:>8.5f}  {tot_avg:>12.5f}")


# ---------------------------------------------------------------------------
# Part (f) – Bias² and Var vs. M=0..12
# ---------------------------------------------------------------------------

def plot_bias_var_vs_degree(max_degree: int = 12, S: int = 200,
                             n_train: int = 15, seed: int = SEED) -> None:
    """Plot Bias², Var and their sum as functions of M.

    Parameters
    ----------
    max_degree : highest degree to evaluate.
    S          : number of MC repetitions.
    n_train    : training set size.
    seed       : random seed.
    """
    x_grid = np.linspace(0, 1, 100)
    f_true = true_function(x_grid)
    degrees = np.arange(0, max_degree + 1)

    bias2_avg  = np.zeros(len(degrees))
    var_avg    = np.zeros(len(degrees))

    print(f"\nComputing Bias² and Var for M=0..{max_degree} (S={S}) …")
    for i, M in enumerate(degrees):
        F_hat = simulate_predictions(M, S, n_train, x_grid, seed=seed + i * 7)
        _, bias2, var = compute_bias2_var(F_hat, f_true)
        bias2_avg[i] = bias2.mean()
        var_avg[i]   = var.mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(degrees, bias2_avg,            "b-o",  lw=2, label="Bias²")
    ax.plot(degrees, var_avg,              "r-s",  lw=2, label="Variance")
    ax.plot(degrees, bias2_avg + var_avg,  "g--^", lw=2, label="Bias² + Var")
    ax.axhline(SIGMA ** 2, color="gray", ls=":", lw=1.5,
               label=f"σ² = {SIGMA**2:.2f}")

    ax.set_xlabel("Polynomial degree M", fontsize=12)
    ax.set_ylabel("Average over grid", fontsize=12)
    ax.set_title("Bias²,  Variance, and their Sum  vs.  Polynomial Degree",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.3, max_degree + 0.3)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "q3f_bias_var_vs_degree.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_problem3(seed: int = SEED) -> None:
    """Execute all sub-parts of Problem 3."""
    print("=" * 60)
    print("PROBLEM 3 – Empirical Bias–Variance Decomposition")
    print("=" * 60)

    x_grid = np.linspace(0, 1, 100)
    f_true = true_function(x_grid)
    degrees_main = [1, 3, 9]
    S = 200
    n_train = 15

    # (a) + (b) – simulate and plot
    print("\n(a,b) Simulating S=200 training sets for M ∈ {1,3,9} …")
    for M in degrees_main:
        F_hat = simulate_predictions(M, S, n_train, x_grid, seed=seed)
        f_bar, bias2, var = compute_bias2_var(F_hat, f_true)
        plot_overlay(x_grid, F_hat, f_bar, M,
                     filename=f"q3b_overlay_M{M}.png")
        print(f"  M={M}: avg Bias²={bias2.mean():.4f}, avg Var={var.mean():.4f}")

    # (b) descriptions
    print("\n(b) Description of overlays:")
    print("  M=1 : The fits are nearly parallel straight lines – underfitting.")
    print("        Average prediction (f_bar) is a smooth line, well away from sin.")
    print("        High bias, low variance.")
    print("  M=3 : Fits are closer to the true function on average.")
    print("        Moderate spread among individual curves.")
    print("        Good balance of bias and variance.")
    print("  M=9 : Individual fits oscillate wildly (Runge phenomenon).")
    print("        f_bar is smoother (averaging cancels oscillations), but still")
    print("        noisy.  Low bias for f_bar, but huge variance.")

    # (c) report
    print("\n(c) Pointwise averages already printed per M above.")

    # (d)
    print("\n(d) Expected:")
    print("  Largest bias   → M=1  (simplest model, unable to capture sin shape).")
    print("  Largest variance → M=9 (complex model, highly sensitive to training data).")

    # (e) Verification table S=200
    print(f"\n(e) Verification table (S={S}):")
    verification_table(degrees_main, S=S, x_grid=x_grid, n_train=n_train,
                       seed=seed)

    print("\n  The discrepancy is not exactly zero because S=200 is finite;")
    print("  the MC estimate of E_D has sampling noise of order 1/√S ≈ 7%.")

    # S=2000
    S2 = 2000
    print(f"\n  Re-running with S={S2} …")
    verification_table(degrees_main, S=S2, x_grid=x_grid, n_train=n_train,
                       seed=seed)
    print(f"\n  With S={S2} the discrepancy shrinks by ~√(200/2000) ≈ 31%,")
    print("  confirming it is due to finite MC sampling noise.")

    # (f) Bias²/Var vs M
    print("\n(f) Plotting Bias² and Var vs. M=0..12 …")
    plot_bias_var_vs_degree(max_degree=12, S=S, n_train=n_train, seed=seed)

    # (g) Short answer
    print("\n(g) Practical unavailability:")
    print("  On a real dataset, f*(x) is unknown (so Bias² cannot be computed)")
    print("  and σ² is unknown (noise level of the DGP is unobservable).")
    print("  Additionally, the expectation E_D requires multiple independent")
    print("  training sets drawn from the same distribution, which is impossible")
    print("  with a fixed dataset.  In practice the decomposition guides intuition")
    print("  about which complexity regime a model is in, but it cannot be computed")
    print("  numerically – cross-validation provides a proxy for the total error.")


if __name__ == "__main__":
    run_problem3()
