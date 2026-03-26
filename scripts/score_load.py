#!/usr/bin/env python3
"""
score_load.py
=============
Compute visual stimulation load scores for NSD/COCO images using
Shannon entropy of the luminance histogram.

Optionally computes supplementary metrics:
  - Edge density (mean Canny edge magnitude)
  - Object count (from COCO instance annotations)

Usage:
    python scripts/score_load.py \
        --nsd_image_dir /path/to/nsd/stimuli/nsd \
        --output data/nsd_image_scores.csv \
        [--coco_instances /path/to/instances_train2017.json] \
        [--workers 8]
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def image_entropy(img_path: str) -> float:
    """Compute Shannon entropy of the luminance histogram."""
    img = Image.open(img_path).convert("L")
    hist, _ = np.histogram(np.array(img).ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # normalize to probability distribution
    return float(scipy_entropy(hist, base=2))


def mean_luminance(img_path: str) -> float:
    """Mean pixel intensity (0-255) of the grayscale image."""
    img = Image.open(img_path).convert("L")
    return float(np.mean(np.array(img)))


def rms_contrast(img_path: str) -> float:
    """RMS contrast: standard deviation of pixel intensities."""
    img = Image.open(img_path).convert("L")
    return float(np.std(np.array(img).astype(np.float64)))


def edge_density(img_path: str) -> float:
    """Mean Canny edge magnitude (requires cv2)."""
    try:
        import cv2
    except ImportError:
        return np.nan
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    edges = cv2.Canny(img, 100, 200)
    return float(np.mean(edges > 0))


def compute_all_metrics(img_path: str) -> dict:
    """Compute all image-level metrics for a single image."""
    basename = os.path.basename(img_path)
    # Extract NSD image index from filename (e.g., 'nsd-00001.png' -> 1)
    # Adjust parsing logic depending on actual NSD filename format
    try:
        nsd_id = int(Path(img_path).stem.split("-")[-1].split("_")[-1])
    except (ValueError, IndexError):
        nsd_id = None

    return {
        "filename": basename,
        "nsd_id": nsd_id,
        "img_path": img_path,
        "entropy": image_entropy(img_path),
        "mean_luminance": mean_luminance(img_path),
        "rms_contrast": rms_contrast(img_path),
        "edge_density": edge_density(img_path),
    }


# ---------------------------------------------------------------------------
# COCO object-count lookup
# ---------------------------------------------------------------------------

def load_coco_object_counts(instances_json: str) -> dict:
    """Return {coco_image_id: object_count} from a COCO instances file."""
    import json

    with open(instances_json, "r") as f:
        data = json.load(f)
    counts: dict[int, int] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        counts[img_id] = counts.get(img_id, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# NSD-to-COCO mapping helper
# ---------------------------------------------------------------------------

def load_nsd_to_coco_mapping(nsd_stim_info_path: str) -> dict:
    """
    Load NSD stimulus info file that maps nsd_id -> coco_id.
    NSD provides this as 'nsd_stim_info_merged.csv' or similar.
    Returns {nsd_id: coco_id}.
    """
    df = pd.read_csv(nsd_stim_info_path)
    # Column names may vary; common names: 'nsdId', 'cocoId'
    nsd_col = [c for c in df.columns if "nsd" in c.lower() and "id" in c.lower()]
    coco_col = [c for c in df.columns if "coco" in c.lower() and "id" in c.lower()]
    if nsd_col and coco_col:
        return dict(zip(df[nsd_col[0]].astype(int), df[coco_col[0]].astype(int)))
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score NSD images on visual stimulation load (entropy)."
    )
    parser.add_argument(
        "--nsd_image_dir",
        required=True,
        help="Directory containing NSD stimulus images.",
    )
    parser.add_argument(
        "--output",
        default="data/nsd_image_scores.csv",
        help="Output CSV path (default: data/nsd_image_scores.csv).",
    )
    parser.add_argument(
        "--coco_instances",
        default=None,
        help="(Optional) Path to COCO instances JSON for object-count metric.",
    )
    parser.add_argument(
        "--nsd_stim_info",
        default=None,
        help="(Optional) Path to NSD stim info CSV for NSD→COCO id mapping.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    args = parser.parse_args()

    # Discover images
    image_dir = Path(args.nsd_image_dir)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(sorted(image_dir.glob(pat)))
    image_paths = [str(p) for p in image_paths]

    if not image_paths:
        print(f"ERROR: No images found in {args.nsd_image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {args.nsd_image_dir}")

    # Compute metrics in parallel
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_all_metrics, p): p for p in image_paths
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rec = future.result()
                records.append(rec)
            except Exception as e:
                print(f"WARNING: Failed on {futures[future]}: {e}", file=sys.stderr)
            if i % 5000 == 0 or i == len(image_paths):
                print(f"  Processed {i}/{len(image_paths)} images")

    df = pd.DataFrame(records)

    # Optionally add COCO object counts
    if args.coco_instances and args.nsd_stim_info:
        print("Loading COCO object counts...")
        coco_counts = load_coco_object_counts(args.coco_instances)
        nsd2coco = load_nsd_to_coco_mapping(args.nsd_stim_info)
        df["coco_id"] = df["nsd_id"].map(nsd2coco)
        df["object_count"] = df["coco_id"].map(coco_counts).fillna(0).astype(int)
    elif args.coco_instances:
        print(
            "WARNING: --coco_instances provided without --nsd_stim_info; "
            "skipping object count.",
            file=sys.stderr,
        )

    # Z-score the entropy (primary load metric)
    mean_ent = df["entropy"].mean()
    std_ent = df["entropy"].std()
    df["load_z"] = (df["entropy"] - mean_ent) / std_ent

    # Z-score supplementary metrics
    for col in ["mean_luminance", "rms_contrast", "edge_density"]:
        if col in df.columns and df[col].notna().sum() > 0:
            m = df[col].mean()
            s = df[col].std()
            if s > 0:
                df[f"{col}_z"] = (df[col] - m) / s

    # Sort and save
    df = df.sort_values("nsd_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved scores for {len(df)} images to {args.output}")
    print(f"  Entropy — mean: {mean_ent:.3f}, std: {std_ent:.3f}")
    print(f"  load_z range: [{df['load_z'].min():.2f}, {df['load_z'].max():.2f}]")


if __name__ == "__main__":
    main()
