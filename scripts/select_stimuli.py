#!/usr/bin/env python3
"""
select_stimuli.py
=================
Select a balanced stimulus set from scored NSD images, sampling uniformly
across the load × abstractness space (3×3 tertile grid).

Ensures:
  - Valid scores on both axes
  - Balanced coverage across the 2D space
  - Orthogonality check (|r| < 0.3 between load and abstractness)

Usage:
    python scripts/select_stimuli.py \
        --scores data/nsd_image_scores.csv \
        --n_per_cell 30 \
        --output data/selected_image_ids.csv \
        [--shared_ids /path/to/shared_1000.csv] \
        [--max_collinearity 0.3] \
        [--seed 42]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def assign_tertile(series: pd.Series) -> pd.Series:
    """Assign each value to a tertile bin (0=low, 1=mid, 2=high)."""
    q33 = series.quantile(1 / 3)
    q66 = series.quantile(2 / 3)
    bins = pd.cut(
        series,
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "mid", "high"],
    )
    return bins


def check_orthogonality(
    load_vals: np.ndarray,
    abstract_vals: np.ndarray,
    threshold: float = 0.3,
) -> tuple[float, bool]:
    """Return (pearson_r, passes_check)."""
    r, p = pearsonr(load_vals, abstract_vals)
    return r, abs(r) < threshold


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factor for each column in X."""
    from sklearn.linear_model import LinearRegression

    vifs = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y_i)
        r2 = reg.score(X_others, y_i)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vifs.append(vif)
    return np.array(vifs)


def main():
    parser = argparse.ArgumentParser(
        description="Select balanced stimulus set from scored NSD images."
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to scored images CSV (must have load_z and abstractness_z columns).",
    )
    parser.add_argument(
        "--n_per_cell",
        type=int,
        default=30,
        help="Target number of images per 3×3 cell (default: 30).",
    )
    parser.add_argument(
        "--output",
        default="data/selected_image_ids.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--shared_ids",
        default=None,
        help="(Optional) CSV/list of NSD shared-1000 image ids to restrict to.",
    )
    parser.add_argument(
        "--max_collinearity",
        type=float,
        default=0.3,
        help="Max |r| between load and abstractness (default: 0.3).",
    )
    parser.add_argument(
        "--max_resample_attempts",
        type=int,
        default=50,
        help="Max resampling attempts if collinearity check fails.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Load scores
    df = pd.read_csv(args.scores)
    print(f"Loaded {len(df)} images from {args.scores}")

    # Filter to valid scores on both axes
    valid_mask = df["load_z"].notna() & df["abstractness_z"].notna()
    df_valid = df[valid_mask].copy()
    print(f"  {len(df_valid)} images with valid load_z and abstractness_z")

    # Optionally restrict to shared set
    if args.shared_ids:
        shared = pd.read_csv(args.shared_ids)
        # Flexible column detection
        id_col = None
        for c in shared.columns:
            if "nsd" in c.lower() or "id" in c.lower():
                id_col = c
                break
        if id_col is None:
            id_col = shared.columns[0]
        shared_set = set(shared[id_col].astype(int))
        df_valid = df_valid[df_valid["nsd_id"].isin(shared_set)]
        print(f"  Restricted to {len(df_valid)} shared-set images")

    if len(df_valid) == 0:
        print("ERROR: No valid images to select from.", file=sys.stderr)
        sys.exit(1)

    # Assign tertiles
    df_valid["load_tertile"] = assign_tertile(df_valid["load_z"])
    df_valid["abstract_tertile"] = assign_tertile(df_valid["abstractness_z"])

    # Report cell sizes
    print("\nAvailable images per cell:")
    cell_counts = df_valid.groupby(["load_tertile", "abstract_tertile"]).size()
    for (lt, at), cnt in cell_counts.items():
        print(f"  load={lt}, abstract={at}: {cnt}")

    # Sampling with collinearity check
    best_selection = None
    best_r = 1.0

    for attempt in range(args.max_resample_attempts):
        selected = []
        for (lt, at), group in df_valid.groupby(
            ["load_tertile", "abstract_tertile"]
        ):
            n_available = len(group)
            n_sample = min(args.n_per_cell, n_available)
            sampled = group.sample(n=n_sample, random_state=rng.integers(1e9))
            selected.append(sampled)

        df_selected = pd.concat(selected, ignore_index=True)
        r, passes = check_orthogonality(
            df_selected["load_z"].values,
            df_selected["abstractness_z"].values,
            args.max_collinearity,
        )

        if abs(r) < abs(best_r):
            best_r = r
            best_selection = df_selected

        if passes:
            print(f"\nOrthogonality check PASSED on attempt {attempt + 1}: r = {r:.4f}")
            break
    else:
        print(
            f"\nWARNING: Could not achieve |r| < {args.max_collinearity} "
            f"after {args.max_resample_attempts} attempts. "
            f"Best r = {best_r:.4f}",
            file=sys.stderr,
        )

    df_selected = best_selection

    # Compute VIF
    X = df_selected[["load_z", "abstractness_z"]].values
    vifs = compute_vif(X)
    print(f"  VIF(load_z) = {vifs[0]:.3f}, VIF(abstractness_z) = {vifs[1]:.3f}")

    # Final summary
    print(f"\nSelected {len(df_selected)} images:")
    final_counts = df_selected.groupby(
        ["load_tertile", "abstract_tertile"]
    ).size()
    for (lt, at), cnt in final_counts.items():
        print(f"  load={lt}, abstract={at}: {cnt}")

    r_final, _ = check_orthogonality(
        df_selected["load_z"].values,
        df_selected["abstractness_z"].values,
    )
    print(f"\nFinal Pearson r(load_z, abstractness_z) = {r_final:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_selected.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
