#!/usr/bin/env python3
"""
voxelwise_regression.py
=======================
Fit voxelwise GLM:
    BOLD_v ~ β1·load + β2·abstract + β3·(load×abstract) + β4·mean_lum + β5·rms_contrast + ε

Produces per-subject beta maps and group-level t-maps (FDR corrected).

Usage:
    python scripts/voxelwise_regression.py \
        --betas results/beta_maps/ \
        --scores data/nsd_image_scores.csv \
        --output results/ \
        [--subjects subj01 subj02 ...] \
        [--fdr_q 0.05]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build design matrix X from scored image DataFrame.
    Returns (X, regressor_names).
    """
    regressors = []
    names = []

    # Primary: load (z-scored entropy)
    regressors.append(df["load_z"].values)
    names.append("load")

    # Primary: abstractness (z-scored)
    regressors.append(df["abstractness_z"].values)
    names.append("abstract")

    # Interaction: load × abstractness
    regressors.append(df["load_z"].values * df["abstractness_z"].values)
    names.append("load_x_abstract")

    # Nuisance: mean luminance (z-scored)
    if "mean_luminance_z" in df.columns:
        regressors.append(df["mean_luminance_z"].values)
        names.append("mean_luminance")
    elif "mean_luminance" in df.columns:
        vals = df["mean_luminance"].values
        regressors.append((vals - vals.mean()) / vals.std())
        names.append("mean_luminance")

    # Nuisance: RMS contrast (z-scored)
    if "rms_contrast_z" in df.columns:
        regressors.append(df["rms_contrast_z"].values)
        names.append("rms_contrast")
    elif "rms_contrast" in df.columns:
        vals = df["rms_contrast"].values
        regressors.append((vals - vals.mean()) / vals.std())
        names.append("rms_contrast")

    X = np.column_stack(regressors)
    return X, names


def fit_voxelwise(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    X: (n_images, n_regressors) design matrix
    Y: (n_images, n_voxels) BOLD responses
    Returns: (n_regressors, n_voxels) beta map
    """
    reg = LinearRegression(fit_intercept=True).fit(X, Y)
    return reg.coef_.T  # (n_regressors, n_voxels)


def fit_voxelwise_with_stats(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OLS fit with t-statistics and p-values for each regressor at each voxel.
    Returns (betas, t_stats, p_values), each of shape (n_regressors, n_voxels).
    """
    n, p = X.shape
    n_voxels = Y.shape[1]

    # Add intercept
    X_int = np.column_stack([np.ones(n), X])
    p_full = X_int.shape[1]

    # OLS: β = (X'X)^-1 X'Y
    XtX_inv = np.linalg.pinv(X_int.T @ X_int)
    betas_full = XtX_inv @ X_int.T @ Y  # (p_full, n_voxels)

    # Residuals
    Y_hat = X_int @ betas_full
    residuals = Y - Y_hat
    dof = n - p_full

    # MSE per voxel
    mse = np.sum(residuals ** 2, axis=0) / dof  # (n_voxels,)

    # Standard error of coefficients
    # se_j = sqrt(mse * (X'X)^-1_jj)
    betas_out = betas_full[1:, :]  # skip intercept, shape (p, n_voxels)
    t_stats = np.zeros_like(betas_out)
    p_values = np.zeros_like(betas_out)

    for j in range(p):
        se = np.sqrt(mse * XtX_inv[j + 1, j + 1])
        se = np.maximum(se, 1e-10)  # avoid division by zero
        t_stats[j, :] = betas_out[j, :] / se
        p_values[j, :] = 2 * stats.t.sf(np.abs(t_stats[j, :]), df=dof)

    return betas_out, t_stats, p_values


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------

def fdr_correction(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns boolean mask of significant tests.
    """
    p_flat = p_values.ravel()
    n = len(p_flat)
    sorted_idx = np.argsort(p_flat)
    sorted_p = p_flat[sorted_idx]

    thresholds = np.arange(1, n + 1) / n * q
    below = sorted_p <= thresholds
    if below.any():
        max_idx = np.max(np.where(below)[0])
        reject = np.zeros(n, dtype=bool)
        reject[sorted_idx[: max_idx + 1]] = True
    else:
        reject = np.zeros(n, dtype=bool)

    return reject.reshape(p_values.shape)


# ---------------------------------------------------------------------------
# Group-level analysis
# ---------------------------------------------------------------------------

def group_level_ttest(beta_maps: list[np.ndarray], fdr_q: float = 0.05):
    """
    One-sample t-test across subjects for each regressor at each voxel.
    beta_maps: list of (n_regressors, n_voxels) arrays (one per subject).
    Returns (group_t, group_p, significant_mask).
    """
    stacked = np.stack(beta_maps)  # (n_subjects, n_regressors, n_voxels)
    n_subjects = stacked.shape[0]
    n_regressors = stacked.shape[1]
    n_voxels = stacked.shape[2]

    t_maps = np.zeros((n_regressors, n_voxels))
    p_maps = np.zeros((n_regressors, n_voxels))

    for r in range(n_regressors):
        t_vals, p_vals = stats.ttest_1samp(stacked[:, r, :], popmean=0, axis=0)
        t_maps[r, :] = t_vals
        p_maps[r, :] = p_vals

    # FDR correction per regressor
    sig_mask = np.zeros_like(p_maps, dtype=bool)
    for r in range(n_regressors):
        sig_mask[r, :] = fdr_correction(p_maps[r, :], q=fdr_q)

    return t_maps, p_maps, sig_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voxelwise regression: BOLD ~ load + abstract + interaction + nuisance."
    )
    parser.add_argument(
        "--betas",
        required=True,
        help="Directory with per-subject beta .npz files.",
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to image scores CSV (must have load_z, abstractness_z, etc.).",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"subj{i:02d}" for i in range(1, 9)],
        help="Subject IDs (default: subj01 through subj08).",
    )
    parser.add_argument(
        "--fdr_q",
        type=float,
        default=0.05,
        help="FDR q-value for group-level correction (default: 0.05).",
    )
    args = parser.parse_args()

    # Load scores
    df_scores = pd.read_csv(args.scores)
    print(f"Loaded {len(df_scores)} image scores")

    os.makedirs(os.path.join(args.output, "beta_maps"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "group"), exist_ok=True)

    all_beta_maps = []
    regressor_names = None

    for subject in args.subjects:
        beta_file = os.path.join(args.betas, f"{subject}_betas.npz")
        if not os.path.exists(beta_file):
            print(f"  Skipping {subject}: {beta_file} not found")
            continue

        print(f"\nProcessing {subject}...")
        data = np.load(beta_file)
        Y = data["betas"]  # (n_images, n_voxels)
        sub_nsd_ids = data["nsd_ids"]
        print(f"  BOLD data: {Y.shape}")

        # Align scores with beta ordering
        df_sub = df_scores[df_scores["nsd_id"].isin(sub_nsd_ids)].copy()
        df_sub = df_sub.set_index("nsd_id").loc[sub_nsd_ids].reset_index()

        # Drop rows with missing scores
        valid_mask = df_sub["load_z"].notna() & df_sub["abstractness_z"].notna()
        if valid_mask.sum() < len(df_sub):
            print(
                f"  Dropping {(~valid_mask).sum()} images with missing scores"
            )
        df_sub = df_sub[valid_mask]
        Y = Y[valid_mask.values]

        # Build design matrix
        X, regressor_names = build_design_matrix(df_sub)
        print(f"  Design matrix: {X.shape} — regressors: {regressor_names}")

        # Fit
        betas, t_stats, p_values = fit_voxelwise_with_stats(X, Y)
        print(f"  Beta map shape: {betas.shape}")

        # Save per-subject
        out_path = os.path.join(args.output, "beta_maps", f"{subject}_glm.npz")
        np.savez_compressed(
            out_path,
            betas=betas,
            t_stats=t_stats,
            p_values=p_values,
            regressor_names=regressor_names,
            nsd_ids=sub_nsd_ids[valid_mask.values],
        )
        print(f"  Saved → {out_path}")

        all_beta_maps.append(betas)

    # Group-level analysis
    if len(all_beta_maps) < 2:
        print("\nWARNING: Need >= 2 subjects for group-level analysis.")
        return

    # Ensure consistent voxel count
    min_voxels = min(b.shape[1] for b in all_beta_maps)
    all_beta_maps = [b[:, :min_voxels] for b in all_beta_maps]

    print(f"\n{'='*60}")
    print(f"Group-level analysis ({len(all_beta_maps)} subjects)")
    print(f"{'='*60}")

    t_maps, p_maps, sig_mask = group_level_ttest(all_beta_maps, fdr_q=args.fdr_q)

    for i, name in enumerate(regressor_names):
        n_sig = sig_mask[i].sum()
        total = sig_mask[i].size
        pct = 100 * n_sig / total if total > 0 else 0
        print(f"  {name}: {n_sig}/{total} voxels significant ({pct:.1f}%)")

    # Save group results
    group_path = os.path.join(args.output, "group", "group_glm.npz")
    np.savez_compressed(
        group_path,
        t_maps=t_maps,
        p_maps=p_maps,
        significant_mask=sig_mask,
        regressor_names=regressor_names,
        n_subjects=len(all_beta_maps),
        fdr_q=args.fdr_q,
    )
    print(f"\nGroup results saved → {group_path}")


if __name__ == "__main__":
    main()
