"""
EC-417 Assignment 1 – Main Runner
===================================
Runs all problems in sequence and produces every figure and table.

Usage
-----
From the repository root:
    python src/run_all.py            # all problems
    python src/run_all.py --q1       # only Problem 1
    python src/run_all.py --q1 --q2  # Problems 1 and 2

All random seeds are fixed so results are exactly reproducible.
"""

import sys
import os
import argparse
import time

# Ensure src/ is on the path when this script is run from the repo root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EC-417 Assignment 1 - run all problems"
    )
    parser.add_argument("--q1", action="store_true", help="Run Problem 1 only")
    parser.add_argument("--q2", action="store_true", help="Run Problem 2 only")
    parser.add_argument("--q3", action="store_true", help="Run Problem 3 only")
    parser.add_argument("--q4", action="store_true", help="Run Problem 4 only")
    args = parser.parse_args()

    # If no specific problem flag given, run all
    run_all = not any([args.q1, args.q2, args.q3, args.q4])

    t_start = time.perf_counter()

    SEP = "=" * 60

    if run_all or args.q1:
        print("\n" + SEP)
        print("  PROBLEM 1 - Vectorization and Pairwise Distances")
        print(SEP)
        from q1_vectorization import verify_and_benchmark
        verify_and_benchmark(N=2000, d=20, seed=42)

    if run_all or args.q2:
        print("\n" + SEP)
        print("  PROBLEM 2 - Polynomial Curve Fitting")
        print(SEP)
        from q2_polynomial_fitting import run_problem2
        run_problem2(seed=42)

    if run_all or args.q3:
        print("\n" + SEP)
        print("  PROBLEM 3 - Empirical Bias-Variance Decomposition")
        print(SEP)
        from q3_bias_variance import run_problem3
        run_problem3(seed=42)

    if run_all or args.q4:
        print("\n" + SEP)
        print("  PROBLEM 4 - EDA: California Housing")
        print(SEP)
        from q4_eda import run_problem4
        run_problem4()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"All tasks completed in {elapsed:.1f} seconds.")
    print(f"Figures saved to: figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
