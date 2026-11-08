"""
EC-417 Assignment 1 – Problem 2: Polynomial Curve Fitting
==========================================================
Demonstrates underfitting and overfitting by fitting polynomials of varying
degree to noisy data drawn from  y = sin(2πx) + ε,  ε ~ N(0, 0.09).

Parts:
  (a) Generate training (n=15) and test (n=1000) sets.
  (b) 2×2 subplot showing fits for M ∈ {0,1,3,9}.
  (c) Train/Test RMSE vs M for n=15.
  (d) Train/Test RMSE vs M for n=100.
  (e) Which quantity changes most when n goes from 15 to 100?
  (f) Why is the n=15, M=9 training RMSE ≈ 0 but test RMSE large?
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
RNG  = np.random.default_rng(SEED)

SIGMA = 0.3
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def true_function(x: np.ndarray) -> np.ndarray:
    """Ground-truth function f*(x) = sin(2πx).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input values.

    Returns
    -------
    np.ndarray, shape (N,)
        f*(x) values.
    """
    return np.sin(2 * np.pi * x)


def generate_data(n: int, rng: np.random.Generator,
                  sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points from  y = sin(2πx) + N(0, sigma²).

    Parameters
    ----------
    n : int
        Number of samples.
    rng : np.random.Generator
        Seeded random generator.
    sigma : float
        Standard deviation of the noise.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Inputs drawn from Uniform(0,1).
    y : np.ndarray, shape (n,)
        Noisy targets.
    """
    x = rng.uniform(0, 1, n)
    y = true_function(x) + rng.normal(0, sigma, n)
    return x, y


def fit_polynomial(x_train: np.ndarray, y_train: np.ndarray,
                   degree: int) -> np.poly1d:
    """Fit a polynomial of given degree to the training data.

    Parameters
    ----------
    x_train : np.ndarray, shape (n,)
        Training inputs.
    y_train : np.ndarray, shape (n,)
        Training targets.
    degree : int
        Polynomial degree.

    Returns
    -------
    poly : np.poly1d
        Fitted polynomial object.
    """
    coeffs = np.polyfit(x_train, y_train, degree)
    return np.poly1d(coeffs)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Part (a) – Generate data
# ---------------------------------------------------------------------------

def generate_train_test(n_train: int = 15, n_test: int = 1000,
                        seed: int = SEED
                        ) -> tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """Generate training and test splits.

    Parameters
    ----------
    n_train : int
        Number of training points.
    n_test : int
        Number of test points.
    seed : int
        Master random seed.

    Returns
    -------
    x_train, y_train, x_test, y_test : four np.ndarray objects.
    """
    rng = np.random.default_rng(seed)
    x_train, y_train = generate_data(n_train, rng)
    x_test,  y_test  = generate_data(n_test,  rng)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Part (b) – 2×2 subplot for M ∈ {0,1,3,9}
# ---------------------------------------------------------------------------

def plot_polynomial_fits(x_train: np.ndarray, y_train: np.ndarray,
                         degrees: list[int] = None,
                         filename: str = "q2b_polynomial_fits.png") -> None:
    """Produce a 2×2 figure of polynomial fits for specified degrees.

    Parameters
    ----------
    x_train : np.ndarray, shape (n,)
        Training inputs.
    y_train : np.ndarray, shape (n,)
        Training targets.
    degrees : list of int
        Four polynomial degrees to display.
    filename : str
        Output figure filename (saved to figures/).
    """
    if degrees is None:
        degrees = [0, 1, 3, 9]

    x_plot = np.linspace(0, 1, 500)
    f_true = true_function(x_plot)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Polynomial Curve Fitting – Underfitting and Overfitting",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    for ax, M in zip(axes.flat, degrees):
        poly = fit_polynomial(x_train, y_train, M)
        y_fit = poly(x_plot)

        ax.scatter(x_train, y_train, color="steelblue", s=40,
                   zorder=5, label="Training data", alpha=0.85)
        ax.plot(x_plot, f_true, "k-", lw=2, label=r"$f^*(x)=\sin(2\pi x)$")
        ax.plot(x_plot, y_fit, color="crimson", lw=2,
                label=f"Degree-{M} fit")

        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"M = {M}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Part (c) & (d) – RMSE curves
# ---------------------------------------------------------------------------

def compute_rmse_curves(x_train: np.ndarray, y_train: np.ndarray,
                        x_test:  np.ndarray, y_test:  np.ndarray,
                        max_degree: int = 12
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute training and test RMSE for degrees 0…max_degree.

    Parameters
    ----------
    x_train, y_train : training split
    x_test, y_test   : test split
    max_degree       : highest polynomial degree to evaluate

    Returns
    -------
    degrees         : np.ndarray of ints, shape (max_degree+1,)
    train_rmse_arr  : np.ndarray, shape (max_degree+1,)
    test_rmse_arr   : np.ndarray, shape (max_degree+1,)
    """
    degrees = np.arange(0, max_degree + 1)
    train_rmse_arr = np.zeros(len(degrees))
    test_rmse_arr  = np.zeros(len(degrees))

    for i, M in enumerate(degrees):
        poly = fit_polynomial(x_train, y_train, M)
        train_rmse_arr[i] = rmse(y_train, poly(x_train))
        test_rmse_arr[i]  = rmse(y_test,  poly(x_test))

    return degrees, train_rmse_arr, test_rmse_arr


def plot_rmse_curves(degrees: np.ndarray,
                     train_rmse: np.ndarray,
                     test_rmse:  np.ndarray,
                     n_train: int,
                     filename: str) -> None:
    """Plot train and test RMSE against polynomial degree.

    Parameters
    ----------
    degrees    : degree values on x-axis.
    train_rmse : training RMSE values.
    test_rmse  : test RMSE values.
    n_train    : training set size (for axis label / title).
    filename   : output filename in figures/.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(degrees, train_rmse, "o-", color="steelblue",
            lw=2, label="Train RMSE")
    ax.plot(degrees, test_rmse,  "s--", color="crimson",
            lw=2, label="Test RMSE")

    best_test = int(degrees[np.argmin(test_rmse)])
    ax.axvline(best_test, color="forestgreen", ls=":", lw=1.5,
               label=f"Best test M={best_test}")

    best_train = int(degrees[np.argmin(train_rmse)])

    ax.set_xlabel("Polynomial degree M", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(f"Train vs Test RMSE  (n = {n_train})", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(f"Min test RMSE at M={best_test}\n"
                f"Min train RMSE at M={best_train}",
                xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print statistics
    best_test_rmse  = test_rmse[best_test]
    best_train_rmse = train_rmse[best_train]
    gaps = test_rmse - train_rmse
    max_gap_M = int(degrees[np.argmax(gaps)])

    print(f"\n  n={n_train}: M minimising test RMSE  = {best_test}  "
          f"(test RMSE={best_test_rmse:.4f})")
    print(f"  n={n_train}: M minimising train RMSE = {best_train} "
          f"(train RMSE={best_train_rmse:.4f})")
    print(f"  n={n_train}: Largest generalisation gap at M={max_gap_M} "
          f"(gap={gaps[max_gap_M]:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_problem2(seed: int = SEED) -> None:
    """Execute all sub-parts of Problem 2."""
    print("=" * 60)
    print("PROBLEM 2 – Polynomial Curve Fitting")
    print("=" * 60)

    # (a) n=15 train, n=1000 test
    x_tr15, y_tr15, x_te, y_te = generate_train_test(15, 1000, seed)
    print(f"\n(a) Training set: n=15, Test set: n=1000")

    # (b) 2×2 subplot
    print("\n(b) Generating 2×2 subplot of polynomial fits …")
    plot_polynomial_fits(x_tr15, y_tr15, degrees=[0, 1, 3, 9],
                         filename="q2b_polynomial_fits.png")

    # (c) RMSE curves for n=15
    print("\n(c) RMSE curves for n=15 …")
    degs, tr_rmse15, te_rmse15 = compute_rmse_curves(
        x_tr15, y_tr15, x_te, y_te, max_degree=12)
    plot_rmse_curves(degs, tr_rmse15, te_rmse15,
                     n_train=15, filename="q2c_rmse_n15.png")

    # (d) n=100
    print("\n(d) RMSE curves for n=100 …")
    rng100 = np.random.default_rng(seed + 1)
    x_tr100, y_tr100 = generate_data(100, rng100)
    degs, tr_rmse100, te_rmse100 = compute_rmse_curves(
        x_tr100, y_tr100, x_te, y_te, max_degree=12)
    plot_rmse_curves(degs, tr_rmse100, te_rmse100,
                     n_train=100, filename="q2d_rmse_n100.png")

    # Comparison note
    print("\n(d) Explanation:")
    print("  With n=100 the training RMSE rises (more data → harder to interpolate)")
    print("  and the test RMSE falls substantially, especially at high M.")
    print("  Variance (sensitivity to training set) decreases with more data,")
    print("  while bias is unchanged.  The two curves converge faster.")

    # (e)
    print("\n(e) Which quantity changes most when n goes from 15→100?")
    print("  Answer: VARIANCE.  Bias depends on model complexity (M), not n.")
    print("  Noise (σ²) is a property of the data-generating process, also fixed.")
    print("  More training data shrinks the variance of the fitted polynomial,")
    print("  reducing overfitting at high degrees.")

    # (f)
    print("\n(f) Why is train RMSE≈0 at M=9, n=15, yet test RMSE is large?")
    print("  A degree-9 polynomial has 10 free parameters; with only 15 points")
    print("  it can nearly interpolate the training set (pass through every")
    print("  point), driving training RMSE toward zero.  But the fit oscillates")
    print("  wildly between the training points (Runge phenomenon), yielding")
    print("  high error on unseen test data.  Near-zero training RMSE does NOT")
    print("  imply a good model – it signals overfitting / high variance.")


if __name__ == "__main__":
    run_problem2()
