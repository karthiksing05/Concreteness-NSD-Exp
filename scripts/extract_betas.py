#!/usr/bin/env python3
"""
extract_betas.py
================
Extract NSD single-trial beta estimates for selected images.

Supports both fsaverage surface and MNI volume formats.
Averages betas across repeated presentations of the same image.

Usage:
    python scripts/extract_betas.py \
        --image_ids data/selected_image_ids.csv \
        --nsd_root /path/to/nsd \
        --output results/beta_maps/ \
        [--subjects subj01 subj02 ...] \
        [--space fsaverage] \
        [--fd_threshold 0.9]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


# ---------------------------------------------------------------------------
# NSD experiment design helpers
# ---------------------------------------------------------------------------

def load_expdesign(nsd_root: str) -> dict:
    """
    Load NSD experiment design from expdesign.mat.
    Returns dict with:
        - 'masterordering': (n_trials,) array — 0-indexed image ID for each trial
        - 'subjectim': (n_subjects, n_images) — which images each subject saw
        - 'sharedix': shared image indices
    """
    mat_path = os.path.join(nsd_root, "nsddata", "experiments", "nsd", "nsd_expdesign.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Cannot find {mat_path}. Ensure --nsd_root points to the NSD data root."
        )
    mat = loadmat(mat_path, squeeze_me=True)
    return {
        "masterordering": mat["masterordering"].astype(int),  # 1-indexed NSD IDs
        "subjectim": mat.get("subjectim", None),
        "sharedix": mat.get("sharedix", None),
    }


def get_trial_indices_for_image(
    subject_idx: int,
    nsd_id: int,
    expdesign: dict,
    n_sessions: int = 40,
    trials_per_session: int = 750,
) -> list[int]:
    """
    Find all trial indices (0-based) where a given subject saw a given image.
    NSD masterordering is a flat list of 1-indexed NSD image IDs, with
    each subject seeing a contiguous block.
    """
    # Subject's trial range in the masterordering
    start = subject_idx * n_sessions * trials_per_session
    end = start + n_sessions * trials_per_session
    sub_ordering = expdesign["masterordering"][start:end]

    # NSD IDs in masterordering are 1-indexed
    trial_indices = np.where(sub_ordering == nsd_id)[0].tolist()
    return trial_indices


# ---------------------------------------------------------------------------
# Beta loading
# ---------------------------------------------------------------------------

def load_beta_fsaverage(
    nsd_root: str,
    subject: str,
    session: int,
    trial_in_session: int,
    hemi: str = "lh",
) -> np.ndarray | None:
    """
    Load a single-trial beta from NSD fsaverage surface space.
    Returns 1D array of shape (n_vertices,) or None if not found.
    """
    # NSD beta files are organized by session
    beta_dir = os.path.join(
        nsd_root,
        "nsddata_betas",
        "ppdata",
        subject,
        "fsaverage",
        "betas_assumehrf",
    )
    # Betas stored as .mgh for fsaverage surface space
    beta_file = os.path.join(beta_dir, f"{hemi}.betas_session{session:02d}.mgh")
    if not os.path.exists(beta_file):
        return None
    img = nib.load(beta_file)
    data = img.get_fdata()
    # .mgh format: shape is (n_vertices, 1, 1, n_trials)
    if data.ndim == 4:
        data = data[:, 0, 0, :]  # (n_vertices, n_trials)
        if trial_in_session < data.shape[1]:
            return data[:, trial_in_session].astype(np.float32)
    elif data.ndim == 2:
        if trial_in_session < data.shape[-1]:
            return data[:, trial_in_session].astype(np.float32)
        elif trial_in_session < data.shape[0]:
            return data[trial_in_session, :].astype(np.float32)
    return None


def load_beta_volume(
    nsd_root: str,
    subject: str,
    session: int,
    trial_in_session: int,
) -> np.ndarray | None:
    """
    Load a single-trial beta from NSD MNI volume space.
    Returns 1D flattened array or None.
    """
    beta_dir = os.path.join(
        nsd_root,
        "nsddata_betas",
        "ppdata",
        subject,
        "func1mm",
        "betas_assumehrf",
    )
    beta_file = os.path.join(beta_dir, f"betas_session{session:02d}.nii.gz")
    if not os.path.exists(beta_file):
        return None
    img = nib.load(beta_file)
    data = img.get_fdata()
    # data shape: (x, y, z, n_trials_in_session)
    if data.ndim == 4 and trial_in_session < data.shape[3]:
        return data[:, :, :, trial_in_session].ravel().astype(np.float32)
    return None


def load_roi_mask(
    nsd_root: str,
    subject: str,
    roi_name: str = "nsdgeneral",
    space: str = "fsaverage",
    hemi: str = "lh",
) -> np.ndarray:
    """
    Load an ROI mask from NSD.
    Returns boolean array.
    """
    if space == "fsaverage":
        label_dir = os.path.join(
            nsd_root,
            "nsddata",
            "freesurfer",
            subject,
            "label",
        )
        mask_file = os.path.join(label_dir, f"{hemi}.{roi_name}.mgz")
        if not os.path.exists(mask_file):
            # Try .nii.gz
            mask_file = os.path.join(label_dir, f"{hemi}.{roi_name}.nii.gz")
        if os.path.exists(mask_file):
            img = nib.load(mask_file)
            return img.get_fdata().ravel() > 0
    else:
        # Volume space
        roi_dir = os.path.join(
            nsd_root, "nsddata", "ppdata", subject, "func1mm", "roi"
        )
        mask_file = os.path.join(roi_dir, f"{roi_name}.nii.gz")
        if os.path.exists(mask_file):
            img = nib.load(mask_file)
            return img.get_fdata().ravel() > 0

    raise FileNotFoundError(f"ROI mask not found: {roi_name} for {subject} in {space}")


# ---------------------------------------------------------------------------
# Motion exclusion
# ---------------------------------------------------------------------------

def load_framewise_displacement(
    nsd_root: str, subject: str
) -> dict[int, np.ndarray] | None:
    """
    Load framewise displacement values per session.
    Returns {session: fd_array} or None if not available.
    """
    fd_dir = os.path.join(nsd_root, "nsddata", "ppdata", subject, "behav")
    if not os.path.isdir(fd_dir):
        return None
    fd_data = {}
    for f in sorted(os.listdir(fd_dir)):
        if "fd" in f.lower() and f.endswith(".tsv"):
            session = int("".join(filter(str.isdigit, f)) or 0)
            fd_data[session] = pd.read_csv(
                os.path.join(fd_dir, f), sep="\t", header=None
            ).values.ravel()
    return fd_data if fd_data else None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_betas_for_subject(
    subject: str,
    subject_idx: int,
    nsd_ids: list[int],
    expdesign: dict,
    nsd_root: str,
    space: str = "fsaverage",
    fd_threshold: float = 0.9,
    trials_per_session: int = 750,
) -> dict[int, np.ndarray]:
    """
    Extract and average betas for a list of NSD image IDs for one subject.
    Returns {nsd_id: averaged_beta_vector}.
    """
    results = {}

    for nsd_id in nsd_ids:
        trial_indices = get_trial_indices_for_image(
            subject_idx, nsd_id, expdesign
        )
        if not trial_indices:
            continue

        betas = []
        for trial_idx in trial_indices:
            session = trial_idx // trials_per_session + 1
            trial_in_session = trial_idx % trials_per_session

            if space == "fsaverage":
                # Load both hemispheres and concatenate
                lh_beta = load_beta_fsaverage(
                    nsd_root, subject, session, trial_in_session, "lh"
                )
                rh_beta = load_beta_fsaverage(
                    nsd_root, subject, session, trial_in_session, "rh"
                )
                if lh_beta is not None and rh_beta is not None:
                    beta = np.concatenate([lh_beta, rh_beta])
                    betas.append(beta)
            else:
                beta = load_beta_volume(
                    nsd_root, subject, session, trial_in_session
                )
                if beta is not None:
                    betas.append(beta)

        if betas:
            # Average across repetitions for SNR improvement
            results[nsd_id] = np.mean(betas, axis=0)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract NSD betas for selected images."
    )
    parser.add_argument(
        "--image_ids",
        required=True,
        help="CSV with selected image IDs (must have 'nsd_id' column).",
    )
    parser.add_argument(
        "--nsd_root", required=True, help="Root path of NSD data download."
    )
    parser.add_argument(
        "--output",
        default="results/beta_maps/",
        help="Output directory for beta arrays.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"subj{i:02d}" for i in range(1, 9)],
        help="Subject IDs (default: subj01 through subj08).",
    )
    parser.add_argument(
        "--space",
        choices=["fsaverage", "volume"],
        default="fsaverage",
        help="Brain space (default: fsaverage).",
    )
    parser.add_argument(
        "--fd_threshold",
        type=float,
        default=0.9,
        help="Framewise displacement threshold in mm (default: 0.9).",
    )
    args = parser.parse_args()

    if nib is None:
        print("ERROR: nibabel is required. pip install nibabel", file=sys.stderr)
        sys.exit(1)

    # Load selected image IDs
    df_ids = pd.read_csv(args.image_ids)
    nsd_ids = sorted(df_ids["nsd_id"].astype(int).unique().tolist())
    print(f"Extracting betas for {len(nsd_ids)} images")

    # Load experiment design
    expdesign = load_expdesign(args.nsd_root)
    print("Loaded NSD experiment design")

    os.makedirs(args.output, exist_ok=True)

    for sub_idx, subject in enumerate(args.subjects):
        print(f"\nProcessing {subject}...")
        sub_betas = extract_betas_for_subject(
            subject=subject,
            subject_idx=sub_idx,
            nsd_ids=nsd_ids,
            expdesign=expdesign,
            nsd_root=args.nsd_root,
            space=args.space,
            fd_threshold=args.fd_threshold,
        )

        if not sub_betas:
            print(f"  WARNING: No betas extracted for {subject}")
            continue

        # Stack into array: (n_images, n_voxels)
        ordered_ids = sorted(sub_betas.keys())
        beta_matrix = np.stack([sub_betas[nid] for nid in ordered_ids])

        # Save
        out_path = os.path.join(args.output, f"{subject}_betas.npz")
        np.savez_compressed(
            out_path,
            betas=beta_matrix,
            nsd_ids=np.array(ordered_ids),
        )
        print(
            f"  Saved {subject}: {beta_matrix.shape} "
            f"({len(ordered_ids)} images) → {out_path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
