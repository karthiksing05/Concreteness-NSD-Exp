"""
nsd_utils.py
============
Unified data loading utilities for the NSD Visual Load & Abstractness experiment.

Provides:
  - NSD experiment design loading (masterordering, subjectim, sharedix)
  - NSD stimulus info / NSD↔COCO ID mapping
  - ROI mask loading with proper integer-label decoding
  - Beta extraction and averaging across repetitions
  - COCO caption loading
  - Brysbaert concreteness norm loading

All functions use config.py paths by default but accept overrides.
"""

import json
import os
import pickle
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

# Import config — graceful fallback if run standalone
try:
    from config import (
        NSD_ROOT,
        SUBJECTS,
        N_SESSIONS_PER_SUBJECT,
        TRIALS_PER_SESSION,
        NSD_EXPDESIGN_PATH,
        NSD_STIM_INFO_PATH,
        ROI_DEFINITIONS,
        ROI_COLLECTIONS,
        COCO_CAPTIONS_FILE,
        COCO_CAPTIONS_VAL_FILE,
        BRYSBAERT_FILE,
        DATA_DIR,
        beta_path,
        roi_path,
    )
except ImportError:
    # Fallback defaults
    NSD_ROOT = Path("data/nsd")
    DATA_DIR = Path("data")
    SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
    N_SESSIONS_PER_SUBJECT = 40
    TRIALS_PER_SESSION = 750
    NSD_EXPDESIGN_PATH = "nsddata/experiments/nsd/nsd_expdesign.mat"
    NSD_STIM_INFO_PATH = "nsddata/experiments/nsd/nsd_stim_info_merged.csv"


# =====================================================================
# Experiment Design
# =====================================================================

_expdesign_cache = {}


def load_expdesign(nsd_root: Path | str | None = None) -> dict:
    """
    Load NSD experiment design from nsd_expdesign.mat.

    Returns dict with:
        masterordering : (n_total_trials,) int array — 1-indexed NSD image IDs
                         for ALL subjects concatenated (subj01 first, then subj02, etc.)
        subjectim      : (n_subjects, n_images_per_subject) — which images each subject saw
        sharedix       : (n_shared,) — 1-indexed NSD IDs of the shared-1000 images
    """
    nsd_root = Path(nsd_root) if nsd_root else NSD_ROOT
    mat_path = nsd_root / NSD_EXPDESIGN_PATH
    cache_key = str(mat_path)

    if cache_key in _expdesign_cache:
        return _expdesign_cache[cache_key]

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Experiment design not found at {mat_path}. "
            f"Run: python scripts/setup_data.py --expdesign"
        )

    mat = loadmat(str(mat_path), squeeze_me=True)
    result = {
        "masterordering": mat["masterordering"].astype(int),
        "subjectim": mat.get("subjectim", None),
        "sharedix": mat.get("sharedix", None),
    }
    _expdesign_cache[cache_key] = result
    return result


def get_shared_image_ids(nsd_root: Path | str | None = None) -> np.ndarray:
    """Return the 1-indexed NSD IDs of the shared-1000 images."""
    expdesign = load_expdesign(nsd_root)
    return expdesign["sharedix"]


# =====================================================================
# NSD ↔ COCO Mapping
# =====================================================================

_stim_info_cache = {}


def load_stim_info(nsd_root: Path | str | None = None) -> pd.DataFrame:
    """
    Load NSD stimulus info (maps nsd_id → coco_id and other metadata).

    Returns DataFrame with columns including 'nsdId' and 'cocoId'.
    """
    nsd_root = Path(nsd_root) if nsd_root else NSD_ROOT
    cache_key = str(nsd_root)

    if cache_key in _stim_info_cache:
        return _stim_info_cache[cache_key]

    # Try CSV first, then pickle
    csv_path = nsd_root / NSD_STIM_INFO_PATH
    pkl_path = csv_path.with_suffix(".pkl")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif pkl_path.exists():
        df = pd.read_pickle(pkl_path)
    else:
        raise FileNotFoundError(
            f"Stim info not found at {csv_path} or {pkl_path}. "
            f"Run: python scripts/setup_data.py --stim-info"
        )

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "nsd" in cl and "id" in cl:
            col_map[c] = "nsdId"
        elif "coco" in cl and "id" in cl:
            col_map[c] = "cocoId"
        elif "coco" in cl and "split" in cl:
            col_map[c] = "cocoSplit"
    df = df.rename(columns=col_map)

    _stim_info_cache[cache_key] = df
    return df


def nsd_to_coco(nsd_root: Path | str | None = None) -> dict:
    """Return {nsd_id (1-indexed): coco_image_id} mapping."""
    df = load_stim_info(nsd_root)
    return dict(zip(df["nsdId"].astype(int), df["cocoId"].astype(int)))


def coco_to_nsd(nsd_root: Path | str | None = None) -> dict:
    """Return {coco_image_id: nsd_id (1-indexed)} mapping."""
    df = load_stim_info(nsd_root)
    return dict(zip(df["cocoId"].astype(int), df["nsdId"].astype(int)))


# =====================================================================
# Trial Index Lookup
# =====================================================================

def get_subject_trial_ordering(
    subject: str,
    nsd_root: Path | str | None = None,
) -> np.ndarray:
    """
    Get the trial ordering for a specific subject.

    Returns (n_trials,) array of 1-indexed NSD image IDs,
    where n_trials = N_SESSIONS * TRIALS_PER_SESSION.
    """
    expdesign = load_expdesign(nsd_root)
    sub_idx = SUBJECTS.index(subject)
    start = sub_idx * N_SESSIONS_PER_SUBJECT * TRIALS_PER_SESSION
    end = start + N_SESSIONS_PER_SUBJECT * TRIALS_PER_SESSION
    return expdesign["masterordering"][start:end]


def get_trial_indices_for_image(
    subject: str,
    nsd_id: int,
    nsd_root: Path | str | None = None,
) -> list[tuple[int, int]]:
    """
    Find all (session, trial_in_session) pairs where subject saw nsd_id.

    Returns list of (session_1indexed, trial_in_session_0indexed) tuples.
    """
    ordering = get_subject_trial_ordering(subject, nsd_root)
    flat_indices = np.where(ordering == nsd_id)[0]
    results = []
    for flat_idx in flat_indices:
        session = flat_idx // TRIALS_PER_SESSION + 1
        trial_in_session = flat_idx % TRIALS_PER_SESSION
        results.append((session, trial_in_session))
    return results


def get_images_seen_by_subject(
    subject: str,
    nsd_root: Path | str | None = None,
) -> np.ndarray:
    """Return sorted array of unique 1-indexed NSD IDs seen by subject."""
    ordering = get_subject_trial_ordering(subject, nsd_root)
    return np.unique(ordering)


# =====================================================================
# ROI Mask Loading
# =====================================================================

def load_roi_surface(
    subject: str,
    hemi: str,
    roi_collection: str,
    nsd_root: Path | str | None = None,
) -> np.ndarray:
    """
    Load a surface ROI file (.mgz) and return the integer label array.

    Returns (n_vertices,) array of integer labels.
    """
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")

    nsd_root = Path(nsd_root) if nsd_root else NSD_ROOT
    relpath = roi_path(subject, hemi, roi_collection)
    fpath = nsd_root / relpath

    if not fpath.exists():
        raise FileNotFoundError(
            f"ROI file not found: {fpath}. "
            f"Run: python scripts/setup_data.py --rois"
        )

    img = nib.load(str(fpath))
    data = np.squeeze(img.get_fdata()).astype(int)
    return data


def get_roi_mask(
    subject: str,
    roi_name: str,
    hemi: str | None = None,
    nsd_root: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """
    Get boolean mask(s) for a named ROI.

    Uses ROI_DEFINITIONS to look up which collection file and integer label
    correspond to the requested ROI name.

    Args:
        subject: NSD subject ID (e.g. 'subj01')
        roi_name: ROI name from ROI_DEFINITIONS (e.g. 'V1v', 'PPA', 'EVC')
        hemi: 'lh', 'rh', or None for both
        nsd_root: override NSD data root

    Returns:
        Dict {'lh': bool_array, 'rh': bool_array} or single hemi if specified.
    """
    if roi_name not in ROI_DEFINITIONS:
        raise ValueError(
            f"Unknown ROI '{roi_name}'. "
            f"Available: {sorted(ROI_DEFINITIONS.keys())}"
        )

    collection, label_spec = ROI_DEFINITIONS[roi_name]
    hemis = [hemi] if hemi else ["lh", "rh"]
    result = {}

    for h in hemis:
        label_data = load_roi_surface(subject, h, collection, nsd_root)

        if label_spec is None:
            # Binary mask (e.g., nsdgeneral): any value > 0
            mask = label_data > 0
        elif isinstance(label_spec, list):
            # Union of multiple labels (e.g., V1 = V1v + V1d)
            mask = np.isin(label_data, label_spec)
        else:
            # Single integer label
            mask = label_data == label_spec

        result[h] = mask

    return result


def get_roi_mask_combined(
    subject: str,
    roi_name: str,
    nsd_root: Path | str | None = None,
) -> np.ndarray:
    """
    Get a single boolean mask by concatenating lh + rh.

    Returns (n_lh_vertices + n_rh_vertices,) boolean array.
    """
    masks = get_roi_mask(subject, roi_name, nsd_root=nsd_root)
    return np.concatenate([masks["lh"], masks["rh"]])


def get_roi_vertex_count(
    subject: str,
    roi_name: str,
    nsd_root: Path | str | None = None,
) -> dict:
    """Return {'lh': n_vertices, 'rh': n_vertices, 'total': n_total}."""
    masks = get_roi_mask(subject, roi_name, nsd_root=nsd_root)
    lh_n = int(masks["lh"].sum())
    rh_n = int(masks["rh"].sum())
    return {"lh": lh_n, "rh": rh_n, "total": lh_n + rh_n}


def list_available_rois(
    subject: str,
    nsd_root: Path | str | None = None,
) -> dict[str, int]:
    """List available ROIs and their vertex counts for a subject."""
    result = {}
    for roi_name in ROI_DEFINITIONS:
        try:
            counts = get_roi_vertex_count(subject, roi_name, nsd_root)
            result[roi_name] = counts["total"]
        except FileNotFoundError:
            pass
    return result


# =====================================================================
# Beta Loading
# =====================================================================

def load_single_beta(
    subject: str,
    session: int,
    trial_in_session: int,
    hemi: str,
    nsd_root: Path | str | None = None,
) -> np.ndarray | None:
    """
    Load a single-trial beta estimate from fsaverage surface space.

    Args:
        subject: e.g. 'subj01'
        session: 1-indexed session number
        trial_in_session: 0-indexed trial within session
        hemi: 'lh' or 'rh'
        nsd_root: override NSD data root

    Returns:
        (n_vertices,) float32 array, or None if file not found.
    """
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")

    nsd_root = Path(nsd_root) if nsd_root else NSD_ROOT
    relpath = beta_path(subject, hemi, session)
    fpath = nsd_root / relpath

    if not fpath.exists():
        return None

    img = nib.load(str(fpath))
    data = img.get_fdata()  # .mgh: shape is (n_vertices, 1, 1, n_trials)

    # .mgh format: (n_vertices, 1, 1, n_trials) — squeeze middle dims
    if data.ndim == 4:
        # Standard .mgh beta shape: (n_vertices, 1, 1, 750)
        data = data[:, 0, 0, :]  # (n_vertices, n_trials)
        if trial_in_session < data.shape[1]:
            return data[:, trial_in_session].astype(np.float32)
    elif data.ndim == 1:
        return data.astype(np.float32)
    elif data.ndim == 2:
        if trial_in_session < data.shape[-1]:
            return data[:, trial_in_session].astype(np.float32)
        elif trial_in_session < data.shape[0]:
            return data[trial_in_session, :].astype(np.float32)

    print(f"  WARNING: Unexpected beta shape {data.shape} for "
          f"{subject}/{hemi}/session{session}/trial{trial_in_session}")
    return None


def extract_betas_for_image(
    subject: str,
    nsd_id: int,
    nsd_root: Path | str | None = None,
) -> np.ndarray | None:
    """
    Extract and average betas across repetitions for a single image.

    Returns (n_lh_vertices + n_rh_vertices,) averaged beta vector,
    or None if no betas found.
    """
    trials = get_trial_indices_for_image(subject, nsd_id, nsd_root)
    if not trials:
        return None

    betas = []
    for session, trial_in_session in trials:
        lh = load_single_beta(subject, session, trial_in_session, "lh", nsd_root)
        rh = load_single_beta(subject, session, trial_in_session, "rh", nsd_root)
        if lh is not None and rh is not None:
            betas.append(np.concatenate([lh, rh]))

    if not betas:
        return None

    # Average across repetitions for SNR improvement
    return np.mean(betas, axis=0).astype(np.float32)


def extract_betas_for_images(
    subject: str,
    nsd_ids: list[int],
    nsd_root: Path | str | None = None,
    roi_name: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract betas for multiple images, optionally masked to an ROI.

    Args:
        subject: NSD subject ID
        nsd_ids: list of 1-indexed NSD image IDs
        nsd_root: override NSD data root
        roi_name: if provided, apply ROI mask from ROI_DEFINITIONS
        verbose: print progress

    Returns:
        (betas, valid_ids) where:
          betas: (n_valid_images, n_voxels) array
          valid_ids: list of nsd_ids that had valid betas
    """
    roi_mask = None
    if roi_name:
        roi_mask = get_roi_mask_combined(subject, roi_name, nsd_root)

    results = {}
    for i, nsd_id in enumerate(nsd_ids):
        beta = extract_betas_for_image(subject, nsd_id, nsd_root)
        if beta is not None:
            if roi_mask is not None:
                # Ensure mask length matches
                if len(roi_mask) == len(beta):
                    beta = beta[roi_mask]
                else:
                    print(f"  WARNING: ROI mask length {len(roi_mask)} != "
                          f"beta length {len(beta)} for {subject}")
            results[nsd_id] = beta

        if verbose and (i + 1) % 50 == 0:
            print(f"  {subject}: {i+1}/{len(nsd_ids)} images processed "
                  f"({len(results)} valid)")

    if not results:
        return np.array([]), []

    valid_ids = sorted(results.keys())
    betas = np.stack([results[nid] for nid in valid_ids])

    if verbose:
        print(f"  {subject}: {betas.shape[0]} images × {betas.shape[1]} voxels")

    return betas, valid_ids


# =====================================================================
# COCO Captions
# =====================================================================

_coco_captions_cache = {}


def load_coco_captions(
    captions_json: Path | str | None = None,
) -> dict[int, list[str]]:
    """
    Load COCO captions.

    Returns {coco_image_id: [caption1, caption2, ...]}.
    """
    paths_to_try = []
    if captions_json:
        paths_to_try.append(Path(captions_json))
    paths_to_try.extend([
        COCO_CAPTIONS_FILE,
        COCO_CAPTIONS_VAL_FILE,
    ])

    all_caps: dict[int, list[str]] = {}

    for p in paths_to_try:
        p = Path(p)
        cache_key = str(p)

        if cache_key in _coco_captions_cache:
            for k, v in _coco_captions_cache[cache_key].items():
                all_caps.setdefault(k, []).extend(v)
            continue

        if not p.exists():
            continue

        with open(p, "r") as f:
            data = json.load(f)

        caps: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            caps.setdefault(img_id, []).append(ann["caption"])

        _coco_captions_cache[cache_key] = caps
        for k, v in caps.items():
            all_caps.setdefault(k, []).extend(v)

    if not all_caps:
        raise FileNotFoundError(
            f"No COCO captions found. Run: python scripts/setup_data.py --coco"
        )

    return all_caps


# =====================================================================
# Brysbaert Concreteness Norms
# =====================================================================

_brysbaert_cache = None


def load_brysbaert(
    path: Path | str | None = None,
) -> dict[str, float]:
    """
    Load Brysbaert et al. (2014) concreteness norms.

    Returns {word_lowercase: concreteness_rating (1-5)}.
    """
    global _brysbaert_cache
    if _brysbaert_cache is not None:
        return _brysbaert_cache

    path = Path(path) if path else BRYSBAERT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Brysbaert norms not found at {path}. "
            f"Run: python scripts/setup_data.py --brysbaert"
        )

    df = pd.read_excel(path)

    # Find the right columns
    word_col = None
    conc_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "word":
            word_col = c
        if "conc" in cl and ("m" in cl or "mean" in cl):
            conc_col = c

    if word_col is None:
        word_col = df.columns[0]
    if conc_col is None:
        for c in df.columns:
            if "conc" in c.lower():
                conc_col = c
                break
    if conc_col is None:
        raise ValueError(f"Cannot find concreteness column in {path}")

    norms = dict(
        zip(
            df[word_col].astype(str).str.lower().str.strip(),
            df[conc_col].astype(float),
        )
    )

    _brysbaert_cache = norms
    return norms


# =====================================================================
# Convenience: Full Pipeline Helpers
# =====================================================================

def get_nsd_image_captions(
    nsd_id: int,
    nsd_root: Path | str | None = None,
    captions_json: Path | str | None = None,
) -> list[str]:
    """Get COCO captions for an NSD image by its NSD ID."""
    mapping = nsd_to_coco(nsd_root)
    coco_id = mapping.get(nsd_id)
    if coco_id is None:
        return []
    caps = load_coco_captions(captions_json)
    return caps.get(coco_id, [])


def verify_data_ready(nsd_root: Path | str | None = None) -> dict[str, bool]:
    """
    Check if all required data files are present.

    Returns dict of {component: is_ready}.
    """
    nsd_root = Path(nsd_root) if nsd_root else NSD_ROOT

    checks = {
        "expdesign": (nsd_root / NSD_EXPDESIGN_PATH).exists(),
        "stim_info": (
            (nsd_root / NSD_STIM_INFO_PATH).exists()
            or (nsd_root / NSD_STIM_INFO_PATH).with_suffix(".pkl").exists()
        ),
        "coco_captions": COCO_CAPTIONS_FILE.exists(),
        "brysbaert": BRYSBAERT_FILE.exists(),
    }

    # Check ROIs for first subject
    roi_ok = True
    for collection in ROI_COLLECTIONS:
        for hemi in ["lh", "rh"]:
            fpath = nsd_root / roi_path("subj01", hemi, collection)
            if not fpath.exists():
                roi_ok = False
                break
        if not roi_ok:
            break
    checks["rois_subj01"] = roi_ok

    return checks


def print_data_status(nsd_root: Path | str | None = None):
    """Print a human-readable data readiness report."""
    status = verify_data_ready(nsd_root)
    print("Data readiness:")
    for component, ready in status.items():
        icon = "[x]" if ready else "[ ]"
        print(f"  {icon} {component}")

    if not all(status.values()):
        missing = [k for k, v in status.items() if not v]
        print(f"\nMissing: {', '.join(missing)}")
        print("Run: python scripts/setup_data.py --all")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NSD data utilities")
    parser.add_argument("--status", action="store_true",
                        help="Check data readiness")
    parser.add_argument("--list-rois", type=str, metavar="SUBJECT",
                        help="List available ROIs for a subject")
    parser.add_argument("--shared-ids", action="store_true",
                        help="Print shared-1000 image IDs")
    args = parser.parse_args()

    if args.status:
        print_data_status()
    elif args.list_rois:
        rois = list_available_rois(args.list_rois)
        print(f"Available ROIs for {args.list_rois}:")
        for name, count in sorted(rois.items()):
            print(f"  {name}: {count} vertices")
    elif args.shared_ids:
        ids = get_shared_image_ids()
        print(f"Shared images: {len(ids)} IDs")
        print(ids[:20], "...")
    else:
        parser.print_help()
