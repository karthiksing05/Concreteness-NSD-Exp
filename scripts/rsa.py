#!/usr/bin/env python3
"""
rsa.py
======
Representational Similarity Analysis (RSA) pipeline.

Compares model RDMs (load, abstractness, combined) with neural RDMs
per ROI per subject. Group-level inference via one-sample t-test on
Fisher-z-transformed Spearman correlations.

Usage:
    python scripts/rsa.py \
        --betas results/beta_maps/ \
        --scores data/nsd_image_scores.csv \
        --output results/rsa_results/ \
        [--nsd_root /path/to/nsd] \
        [--subjects sub-01 sub-02 ...] \
        [--rois nsdgeneral V1v V2v V3v hV4 LOC PPA RSC] \
        [--fdr_q 0.05]
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

try:
    import nibabel as nib
except ImportError:
    nib = None


# ---------------------------------------------------------------------------
# Model RDMs
# ---------------------------------------------------------------------------

def compute_model_rdm(scores: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix from score vectors.
    scores: (n_images, n_dims) — e.g. (n, 2) for [load, abstract].
    Returns: (n_images, n_images) symmetric distance matrix.
    """
    dists = pdist(scores, metric="euclidean")
    return squareform(dists)


def compute_single_dim_rdm(scores_1d: np.ndarray) -> np.ndarray:
    """RDM from a single dimension: absolute pairwise differences."""
    n = len(scores_1d)
    rdm = np.abs(scores_1d[:, None] - scores_1d[None, :])
    return rdm


# ---------------------------------------------------------------------------
# Neural RDM
# ---------------------------------------------------------------------------

def compute_rdm_neural(betas: np.ndarray) -> np.ndarray:
    """
    Compute neural RDM from BOLD patterns.
    betas: (n_images, n_voxels) → RDM: (n_images, n_images)
    Uses 1 - Pearson correlation as dissimilarity.
    """
    corr = np.corrcoef(betas)  # (n_images, n_images)
    return 1 - corr


# ---------------------------------------------------------------------------
# RSA correlation
# ---------------------------------------------------------------------------

def rsa_correlation(model_rdm: np.ndarray, neural_rdm: np.ndarray) -> float:
    """
    Spearman correlation between upper triangles of model and neural RDMs.
    """
    model_vec = squareform(model_rdm, checks=False)
    neural_vec = squareform(neural_rdm, checks=False)
    r, _ = spearmanr(model_vec, neural_vec)
    return float(r)


def fisher_z(r: float) -> float:
    """Fisher z-transformation of a correlation coefficient."""
    r = np.clip(r, -0.9999, 0.9999)
    return float(np.arctanh(r))


# ---------------------------------------------------------------------------
# ROI mask loading
# ---------------------------------------------------------------------------

def load_roi_mask_fsaverage(
    nsd_root: str, subject: str, roi_name: str, hemi: str = "lh"
) -> np.ndarray | None:
    """Load ROI mask from NSD fsaverage labels. Returns boolean array or None."""
    label_dir = os.path.join(
        nsd_root, "nsddata", "freesurfer", subject, "label"
    )
    for ext in [".mgz", ".nii.gz", ".label"]:
        mask_file = os.path.join(label_dir, f"{hemi}.{roi_name}{ext}")
        if os.path.exists(mask_file):
            if ext == ".label":
                # FreeSurfer label file: vertex indices
                vertices = np.loadtxt(mask_file, skiprows=2, usecols=0, dtype=int)
                return vertices
            else:
                img = nib.load(mask_file)
                return img.get_fdata().ravel() > 0
    return None


def get_roi_indices(
    nsd_root: str | None,
    subject: str,
    roi_name: str,
    n_voxels: int,
) -> np.ndarray | None:
    """
    Get boolean mask for an ROI.
    If nsd_root is provided, tries to load from NSD.
    Otherwise, returns None (caller uses full brain).
    """
    if nsd_root is None:
        return None

    # Try both hemispheres for surface space
    for hemi in ["lh", "rh"]:
        mask = load_roi_mask_fsaverage(nsd_root, subject, roi_name, hemi)
        if mask is not None:
            if isinstance(mask, np.ndarray) and mask.dtype == bool:
                if len(mask) <= n_voxels:
                    # Pad to full brain size if needed
                    full_mask = np.zeros(n_voxels, dtype=bool)
                    full_mask[: len(mask)] = mask
                    return full_mask
            elif isinstance(mask, np.ndarray) and mask.dtype in [np.int32, np.int64]:
                # Label format: vertex indices
                full_mask = np.zeros(n_voxels, dtype=bool)
                valid_idx = mask[mask < n_voxels]
                full_mask[valid_idx] = True
                return full_mask

    # Try volume space
    roi_dir = os.path.join(
        nsd_root, "nsddata", "ppdata", subject, "func1mm", "roi"
    )
    for ext in [".nii.gz", ".mgz"]:
        mask_file = os.path.join(roi_dir, f"{roi_name}{ext}")
        if os.path.exists(mask_file):
            img = nib.load(mask_file)
            mask = img.get_fdata().ravel() > 0
            if len(mask) == n_voxels:
                return mask
    return None


# ---------------------------------------------------------------------------
# Group-level RSA inference
# ---------------------------------------------------------------------------

def group_ttest_rsa(
    rsa_values: list[float],
) -> tuple[float, float, float]:
    """
    One-sample t-test on Fisher-z-transformed RSA values.
    Returns (mean_r, t_stat, p_value).
    """
    from scipy.stats import ttest_1samp

    z_vals = [fisher_z(r) for r in rsa_values]
    mean_z = np.mean(z_vals)
    mean_r = np.tanh(mean_z)

    if len(z_vals) < 2:
        return mean_r, np.nan, np.nan

    t_stat, p_val = ttest_1samp(z_vals, popmean=0)
    return float(mean_r), float(t_stat), float(p_val)


def fdr_correction(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean mask."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RSA pipeline for NSD experiment.")
    parser.add_argument(
        "--betas",
        required=True,
        help="Directory with per-subject beta .npz files.",
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to image scores CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/rsa_results/",
        help="Output directory.",
    )
    parser.add_argument(
        "--nsd_root",
        default=None,
        help="NSD data root (for loading ROI masks). If not provided, uses full brain.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"sub-{i:02d}" for i in range(1, 9)],
        help="Subject IDs.",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        default=[
            "nsdgeneral",
            "V1v",
            "V2v",
            "V3v",
            "hV4",
            "LOC",
            "OFA",
            "PPA",
            "RSC",
        ],
        help="ROI names to analyze.",
    )
    parser.add_argument(
        "--fdr_q",
        type=float,
        default=0.05,
        help="FDR q-value for group-level correction.",
    )
    args = parser.parse_args()

    # Load scores
    df_scores = pd.read_csv(args.scores)
    print(f"Loaded {len(df_scores)} image scores")

    os.makedirs(args.output, exist_ok=True)

    # Build model RDMs from scores
    # We'll construct them per subject (matching their image subset)

    # Define model RDM types
    model_names = ["combined", "load_only", "abstract_only"]

    # Results storage
    # {roi: {model: [r_sub1, r_sub2, ...]}}
    results: dict[str, dict[str, list[float]]] = {
        roi: {m: [] for m in model_names} for roi in args.rois
    }

    for subject in args.subjects:
        beta_file = os.path.join(args.betas, f"{subject}_betas.npz")
        if not os.path.exists(beta_file):
            print(f"  Skipping {subject}: {beta_file} not found")
            continue

        print(f"\nProcessing {subject}...")
        data = np.load(beta_file)
        Y = data["betas"]  # (n_images, n_voxels)
        sub_nsd_ids = data["nsd_ids"]
        n_voxels = Y.shape[1]

        # Align scores
        df_sub = df_scores[df_scores["nsd_id"].isin(sub_nsd_ids)].copy()
        df_sub = df_sub.set_index("nsd_id").loc[sub_nsd_ids].reset_index()

        valid_mask = df_sub["load_z"].notna() & df_sub["abstractness_z"].notna()
        df_sub = df_sub[valid_mask]
        Y = Y[valid_mask.values]

        load_scores = df_sub["load_z"].values
        abstract_scores = df_sub["abstractness_z"].values

        # Model RDMs
        combined_rdm = compute_model_rdm(
            np.column_stack([load_scores, abstract_scores])
        )
        load_rdm = compute_single_dim_rdm(load_scores)
        abstract_rdm = compute_single_dim_rdm(abstract_scores)
        model_rdms = {
            "combined": combined_rdm,
            "load_only": load_rdm,
            "abstract_only": abstract_rdm,
        }

        for roi_name in args.rois:
            # Get ROI mask
            roi_mask = get_roi_indices(
                args.nsd_root, subject, roi_name, n_voxels
            )

            if roi_mask is not None:
                roi_betas = Y[:, roi_mask]
                if roi_betas.shape[1] == 0:
                    print(f"    {roi_name}: 0 voxels in mask, skipping")
                    continue
            else:
                # Use full brain if no mask
                roi_betas = Y

            # Neural RDM
            neural_rdm = compute_rdm_neural(roi_betas)

            # RSA for each model
            for model_name in model_names:
                r = rsa_correlation(model_rdms[model_name], neural_rdm)
                results[roi_name][model_name].append(r)

            print(
                f"    {roi_name} ({roi_betas.shape[1]} voxels): "
                f"combined r={results[roi_name]['combined'][-1]:.4f}, "
                f"load r={results[roi_name]['load_only'][-1]:.4f}, "
                f"abstract r={results[roi_name]['abstract_only'][-1]:.4f}"
            )

    # Group-level analysis
    print(f"\n{'='*70}")
    print("Group-level RSA results")
    print(f"{'='*70}")

    group_results = []

    for roi_name in args.rois:
        for model_name in model_names:
            r_values = results[roi_name][model_name]
            if len(r_values) == 0:
                continue
            mean_r, t_stat, p_val = group_ttest_rsa(r_values)
            group_results.append(
                {
                    "roi": roi_name,
                    "model": model_name,
                    "mean_r": mean_r,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "n_subjects": len(r_values),
                }
            )

    df_group = pd.DataFrame(group_results)

    # FDR correction across all ROI × model tests
    if len(df_group) > 0 and df_group["p_value"].notna().any():
        p_vals = df_group["p_value"].fillna(1.0).values
        sig = fdr_correction(p_vals, q=args.fdr_q)
        df_group["significant_fdr"] = sig
    else:
        df_group["significant_fdr"] = False

    # Print results
    print(f"\n{'ROI':<15} {'Model':<18} {'mean_r':>8} {'t':>8} {'p':>10} {'sig':>5}")
    print("-" * 70)
    for _, row in df_group.iterrows():
        sig_str = " *" if row.get("significant_fdr", False) else ""
        print(
            f"{row['roi']:<15} {row['model']:<18} "
            f"{row['mean_r']:>8.4f} {row['t_stat']:>8.3f} "
            f"{row['p_value']:>10.6f}{sig_str}"
        )

    # Save results
    df_group.to_csv(os.path.join(args.output, "rsa_group_results.csv"), index=False)

    # Save per-subject r-values
    per_sub_records = []
    for roi_name in args.rois:
        for model_name in model_names:
            for i, r in enumerate(results[roi_name][model_name]):
                per_sub_records.append(
                    {
                        "roi": roi_name,
                        "model": model_name,
                        "subject_idx": i,
                        "rsa_r": r,
                        "rsa_z": fisher_z(r),
                    }
                )
    df_per_sub = pd.DataFrame(per_sub_records)
    df_per_sub.to_csv(
        os.path.join(args.output, "rsa_per_subject.csv"), index=False
    )

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
