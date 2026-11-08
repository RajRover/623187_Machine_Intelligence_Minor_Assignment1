"""
EC-417 Assignment 1 – Problem 4: Exploratory Data Analysis (California Housing)
================================================================================
Loads the California Housing dataset and performs a disciplined EDA including:
  (a) Shape, dtypes, missing values, summary statistics.
  (b) Target variable histogram + right-censoring anomaly.
  (c) Feature histograms + skewness analysis.
  (d) Correlation matrix heatmap.
  (e) Spatial scatter plot (Latitude vs Longitude) coloured by target.
  (f) Data-leakage demonstration: correct vs. incorrect standardisation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEED = 42
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset as a DataFrame + Series.

    Returns
    -------
    df : pd.DataFrame, shape (20640, 8)
        Feature matrix with named columns.
    target : pd.Series, shape (20640,)
        Median house value (in units of $100,000).
    """
    data = fetch_california_housing(as_frame=True)
    df     = data.data
    target = data.target
    return df, target


# ---------------------------------------------------------------------------
# Part (a) – Basic statistics
# ---------------------------------------------------------------------------

def basic_statistics(df: pd.DataFrame, target: pd.Series) -> None:
    """Print shape, dtypes, missing values, and summary statistics.

    Parameters
    ----------
    df     : feature DataFrame.
    target : target Series.
    """
    full = df.copy()
    full["MedHouseVal"] = target

    print("\n(a) Dataset Overview")
    print("=" * 60)
    print(f"Shape: {full.shape}")
    print(f"\nDtype of each column:")
    print(full.dtypes.to_string())
    print(f"\nMissing values per column:")
    print(full.isnull().sum().to_string())
    print(f"\nSummary statistics:")
    print(full.describe().to_string())


# ---------------------------------------------------------------------------
# Part (b) – Target histogram
# ---------------------------------------------------------------------------

def plot_target_histogram(target: pd.Series) -> None:
    """Plot the target-variable histogram and report the right-censoring anomaly.

    Parameters
    ----------
    target : target Series (median house values in $100k).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(target, bins=80, color="steelblue", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.axvline(target.max(), color="crimson", lw=2, ls="--",
               label=f"Max = {target.max():.4f} ($100k)")

    ax.set_xlabel("Median House Value (×$100,000)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Target Variable – California Housing",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "q4b_target_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\n(b) Target analysis:")
    print(f"  Maximum value: {target.max():.4f}  (i.e., ${target.max()*100_000:,.0f})")
    count_at_max = (target == target.max()).sum()
    print(f"  Records at exact maximum: {count_at_max}")
    print(f"  Phenomenon: RIGHT-CENSORING (top-coding).")
    print(f"  Reason: The original survey instrument capped house values at $500,000.")
    print(f"  All blocks with true value ≥ $500k are recorded as exactly $500k.")
    print(f"  Effect on regression: The model will systematically under-predict")
    print(f"  values for expensive districts because the true distribution of the")
    print(f"  target above $500k is invisible; the model learns a ceiling, not")
    print(f"  the actual relationship between predictors and price in that range.")


# ---------------------------------------------------------------------------
# Part (c) – Feature histograms
# ---------------------------------------------------------------------------

def plot_feature_histograms(df: pd.DataFrame) -> None:
    """Plot histograms of all eight predictor features.

    Parameters
    ----------
    df : feature DataFrame with 8 columns.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Feature Distributions – California Housing",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flat, df.columns):
        ax.hist(df[col], bins=60, color="teal", edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "q4c_feature_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\n(c) Skewed features with implausible maxima:")
    print(f"  1. AveRooms (max={df['AveRooms'].max():.1f}):")
    print(f"     Average rooms per household > 100 is physically impossible for")
    print(f"     a residential census block.  Likely a data-entry error or a")
    print(f"     block with very few households (denominator ≈ 0 inflates the ratio).")
    print(f"     Treatment: Winsorise at e.g. the 99th percentile")
    print(f"     ({df['AveRooms'].quantile(0.99):.1f} rooms) or log-transform.")
    print()
    print(f"  2. AveBedrms (max={df['AveBedrms'].max():.1f}):")
    print(f"     Average bedrooms > 20 per household is implausible for a typical")
    print(f"     residential neighbourhood.  Same small-household-count artefact.")
    print(f"     Treatment: Winsorise or log-transform together with AveRooms.")
    print()
    print(f"  3. AveOccup (max={df['AveOccup'].max():.1f}):")
    print(f"     Average occupants per household > 100 is clearly erroneous; a")
    print(f"     block of one or two households with many occupants produces an")
    print(f"     extreme ratio.  Treatment: Cap at 99th percentile")
    print(f"     ({df['AveOccup'].quantile(0.99):.1f}) — removal is not preferred")
    print(f"     because occupancy correlates with housing pressure.")


# ---------------------------------------------------------------------------
# Part (d) – Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, target: pd.Series) -> None:
    """Display the Pearson correlation matrix as a seaborn heatmap.

    Parameters
    ----------
    df     : feature DataFrame.
    target : target Series.
    """
    full = df.copy()
    full["MedHouseVal"] = target

    corr = full.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    # Show lower triangle + diagonal
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, mask=False,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Pearson Correlation Matrix – California Housing",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "q4d_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    target_corrs = corr["MedHouseVal"].drop("MedHouseVal").abs()
    best_predictor = target_corrs.idxmax()

    # Find strongest inter-predictor correlation
    inter = corr.drop("MedHouseVal", axis=0).drop("MedHouseVal", axis=1)
    inter_abs = inter.where(np.tril(np.ones(inter.shape, dtype=bool), k=-1)).abs()
    i_idx = np.unravel_index(np.nanargmax(inter_abs.values), inter_abs.shape)
    feat_a, feat_b = inter_abs.columns[i_idx[1]], inter_abs.index[i_idx[0]]

    print(f"\n(d) Correlation analysis:")
    print(f"  Strongest predictor for target: {best_predictor}")
    print(f"  (|r| = {corr.loc[best_predictor, 'MedHouseVal']:.3f})")
    print(f"\n  Strongest inter-predictor correlation: {feat_a} ↔ {feat_b}")
    print(f"  (r = {corr.loc[feat_b, feat_a]:.3f})")
    print(f"\n  Multicollinearity between {feat_a} and {feat_b} inflates the")
    print(f"  variance of OLS coefficients and makes feature importances")
    print(f"  individually unreliable – the model cannot separate their effects.")


# ---------------------------------------------------------------------------
# Part (e) – Spatial scatter plot
# ---------------------------------------------------------------------------

def plot_spatial_scatter(df: pd.DataFrame, target: pd.Series) -> None:
    """Scatter plot of Longitude vs Latitude coloured by target value.

    Parameters
    ----------
    df     : feature DataFrame (must contain 'Longitude', 'Latitude').
    target : target Series (median house values).
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(df["Longitude"], df["Latitude"],
                    c=target, cmap="YlOrRd",
                    s=1.2, alpha=0.55, linewidths=0)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Median House Value (×$100k)", fontsize=10)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude",  fontsize=12)
    ax.set_title("California Housing – Spatial Distribution of Median House Values",
                 fontsize=12)

    # Annotate the two high-value clusters
    ax.annotate("San Francisco Bay Area",
                xy=(-122.4, 37.8), xytext=(-120.5, 38.3),
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
                fontsize=9, color="navy", fontweight="bold")
    ax.annotate("Greater Los Angeles",
                xy=(-118.3, 34.1), xytext=(-116.5, 33.2),
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
                fontsize=9, color="navy", fontweight="bold")

    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "q4e_spatial_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\n(e) Spatial analysis:")
    print(f"  Two dense high-value clusters: the San Francisco Bay Area")
    print(f"  (~37°N, 122°W) and Greater Los Angeles (~34°N, 118°W).")
    print(f"\n  Spatial autocorrelation violates i.i.d.: neighbouring census blocks")
    print(f"  share unobserved geographic covariates (school quality, proximity to")
    print(f"  parks, transit) that are not in the feature set.  Their targets are")
    print(f"  therefore correlated, not independent.")
    print(f"\n  Consequence for random train/test split: blocks from the same")
    print(f"  neighbourhood are split across train and test.  The model can")
    print(f"  implicitly 'remember' a neighbourhood's characteristics from the")
    print(f"  training examples nearby, artificially inflating test-set performance")
    print(f"  relative to what would be achieved on a truly unseen geography.")


# ---------------------------------------------------------------------------
# Part (f) – Data leakage
# ---------------------------------------------------------------------------

def data_leakage_demo(df: pd.DataFrame, target: pd.Series) -> None:
    """Demonstrate leakage by comparing correct vs. incorrect standardisation.

    Parameters
    ----------
    df     : feature DataFrame.
    target : target Series.
    """
    X = df.values.astype(np.float64)

    # Split 80/20
    X_train, X_test, _, _ = train_test_split(X, target, test_size=0.2,
                                              random_state=SEED, shuffle=True)

    # --- Method (i): correct – fit scaler on training split only ---
    mu_train  = X_train.mean(axis=0)
    std_train = X_train.std(axis=0, ddof=0)

    X_train_i = (X_train - mu_train) / std_train
    X_test_i  = (X_test  - mu_train) / std_train   # use TRAIN stats on test

    # --- Method (ii): incorrect – fit scaler on full dataset ---
    mu_full  = X.mean(axis=0)
    std_full = X.std(axis=0, ddof=0)

    X_full_scaled = (X - mu_full) / std_full
    n_train = X_train.shape[0]
    # Reproduce the same split indices
    from sklearn.model_selection import train_test_split as tts
    idx = np.arange(len(X))
    idx_tr, idx_te = tts(idx, test_size=0.2, random_state=SEED)

    X_test_ii = X_full_scaled[idx_te]
    X_train_ii = X_full_scaled[idx_tr]

    test_means_i  = X_test_i.mean(axis=0)
    test_means_ii = X_test_ii.mean(axis=0)

    diff = test_means_i - test_means_ii

    print(f"\n(f) Data-leakage demonstration (test-set feature means):")
    print(f"\n{'Feature':<15} {'Method(i)':>12} {'Method(ii)':>12} {'Difference':>12}")
    print("-" * 53)
    for name, m1, m2, d in zip(df.columns, test_means_i, test_means_ii, diff):
        print(f"{name:<15} {m1:>12.6f} {m2:>12.6f} {d:>12.2e}")

    print(f"\n  Explanation: Method (i) is the only CORRECT procedure because")
    print(f"  the test set must be treated as unseen data throughout.  Computing")
    print(f"  statistics on the full dataset leaks information from test examples")
    print(f"  into the scaler (the test mean/std influence the scaling of the")
    print(f"  training features).  Although the difference is small here because")
    print(f"  the dataset is large (n=20,640), in a REALISTIC scenario with a")
    print(f"  small dataset (e.g., n=100) the leakage can be substantial: the")
    print(f"  20% test-set examples materially shift the mean/std estimates,")
    print(f"  resulting in artificially low test errors and over-optimistic")
    print(f"  selection of hyperparameters during cross-validation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_problem4() -> None:
    """Execute all sub-parts of Problem 4."""
    print("=" * 60)
    print("PROBLEM 4 – Exploratory Data Analysis: California Housing")
    print("=" * 60)

    df, target = load_california_housing()

    # (a)
    basic_statistics(df, target)

    # (b)
    print("\n(b) Plotting target histogram …")
    plot_target_histogram(target)

    # (c)
    print("\n(c) Plotting feature histograms …")
    plot_feature_histograms(df)

    # (d)
    print("\n(d) Plotting correlation heatmap …")
    plot_correlation_heatmap(df, target)

    # (e)
    print("\n(e) Plotting spatial scatter …")
    plot_spatial_scatter(df, target)

    # (f)
    print("\n(f) Data leakage demonstration …")
    data_leakage_demo(df, target)


if __name__ == "__main__":
    run_problem4()
