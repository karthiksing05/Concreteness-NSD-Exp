#!/usr/bin/env python3
"""
score_abstractness.py
=====================
Score NSD/COCO images on conceptual abstractness using Brysbaert et al. (2014)
concreteness norms applied to COCO captions.

Abstractness = 5.0 - mean_concreteness (of content words in captions).

Usage:
    python scripts/score_abstractness.py \
        --scores data/nsd_image_scores.csv \
        --brysbaert data/brysbaert_2014_concreteness.xlsx \
        --coco_captions /path/to/coco/annotations/captions_train2017.json \
        [--nsd_stim_info /path/to/nsd_stim_info_merged.csv] \
        [--min_words 3]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------------------------
# Brysbaert concreteness norms
# ---------------------------------------------------------------------------

def load_brysbaert(path: str) -> dict:
    """
    Load Brysbaert et al. (2014) concreteness norms.
    Returns {word: concreteness_rating} dict.
    """
    df = pd.read_excel(path)
    # The file typically has columns: 'Word', 'Conc.M' (mean concreteness)
    # Try to find the right columns flexibly
    word_col = None
    conc_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("word",):
            word_col = c
        if "conc" in cl and ("m" in cl or "mean" in cl):
            conc_col = c
    if word_col is None:
        # Fallback: first string column
        word_col = df.columns[0]
    if conc_col is None:
        # Fallback: search for any numeric column with 'conc' in the name
        for c in df.columns:
            if "conc" in c.lower():
                conc_col = c
                break
    if conc_col is None:
        raise ValueError(
            f"Could not identify concreteness column in {path}. "
            f"Columns found: {list(df.columns)}"
        )

    norms = dict(
        zip(df[word_col].astype(str).str.lower().str.strip(), df[conc_col].astype(float))
    )
    print(f"Loaded {len(norms)} concreteness norms from {path}")
    return norms


# ---------------------------------------------------------------------------
# COCO captions
# ---------------------------------------------------------------------------

def load_coco_captions(captions_json: str) -> dict:
    """
    Load COCO captions file.
    Returns {coco_image_id: [caption_str, ...]}.
    """
    with open(captions_json, "r") as f:
        data = json.load(f)
    caps: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        caps.setdefault(img_id, []).append(ann["caption"])
    print(f"Loaded captions for {len(caps)} COCO images")
    return caps


# ---------------------------------------------------------------------------
# NSD → COCO mapping
# ---------------------------------------------------------------------------

def load_nsd_to_coco(path: str) -> dict:
    """Load NSD stim info and return {nsd_id: coco_id}."""
    df = pd.read_csv(path)
    nsd_col = [c for c in df.columns if "nsd" in c.lower() and "id" in c.lower()]
    coco_col = [c for c in df.columns if "coco" in c.lower() and "id" in c.lower()]
    if nsd_col and coco_col:
        return dict(zip(df[nsd_col[0]].astype(int), df[coco_col[0]].astype(int)))
    # Fallback: assume first two columns
    return dict(zip(df.iloc[:, 0].astype(int), df.iloc[:, 1].astype(int)))


# ---------------------------------------------------------------------------
# Abstractness scoring
# ---------------------------------------------------------------------------

def score_image_abstractness(
    captions: list[str], norms: dict, stops: set, min_words: int = 3
) -> float | None:
    """
    Score a single image's abstractness from its captions.
    Returns abstractness = 5.0 - mean_concreteness, or None if < min_words matched.
    """
    ratings = []
    for cap in captions:
        tokens = [
            w.lower()
            for w in word_tokenize(cap)
            if w.isalpha() and w.lower() not in stops
        ]
        ratings.extend([norms[t] for t in tokens if t in norms])
    if len(ratings) < min_words:
        return None
    mean_concreteness = float(np.mean(ratings))
    return float(5.0 - mean_concreteness)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add abstractness scores to NSD image scores."
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to existing scores CSV (from score_load.py).",
    )
    parser.add_argument(
        "--brysbaert",
        required=True,
        help="Path to Brysbaert concreteness norms (.xlsx).",
    )
    parser.add_argument(
        "--coco_captions",
        required=True,
        help="Path to COCO captions JSON (e.g. captions_train2017.json).",
    )
    parser.add_argument(
        "--nsd_stim_info",
        default=None,
        help="Path to NSD stim info CSV (maps nsd_id → coco_id). "
        "If not provided, expects 'coco_id' column already in scores CSV.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=3,
        help="Minimum matched content words to produce a score (default: 3).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite --scores file).",
    )
    args = parser.parse_args()

    # Load existing scores
    df = pd.read_csv(args.scores)
    print(f"Loaded {len(df)} image records from {args.scores}")

    # Ensure we have coco_id
    if "coco_id" not in df.columns:
        if args.nsd_stim_info is None:
            print(
                "ERROR: 'coco_id' column not found in scores CSV and "
                "--nsd_stim_info not provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        nsd2coco = load_nsd_to_coco(args.nsd_stim_info)
        df["coco_id"] = df["nsd_id"].map(nsd2coco)

    # Load resources
    norms = load_brysbaert(args.brysbaert)
    coco_caps = load_coco_captions(args.coco_captions)
    stops = set(stopwords.words("english"))

    # Score each image
    abstractness_scores = []
    n_scored = 0
    n_skipped = 0
    for _, row in df.iterrows():
        coco_id = row.get("coco_id")
        if pd.isna(coco_id):
            abstractness_scores.append(np.nan)
            n_skipped += 1
            continue
        coco_id = int(coco_id)
        captions = coco_caps.get(coco_id, [])
        if not captions:
            abstractness_scores.append(np.nan)
            n_skipped += 1
            continue
        score = score_image_abstractness(captions, norms, stops, args.min_words)
        if score is None:
            abstractness_scores.append(np.nan)
            n_skipped += 1
        else:
            abstractness_scores.append(score)
            n_scored += 1

    df["abstractness"] = abstractness_scores

    # Z-score abstractness (only for non-NaN)
    valid = df["abstractness"].notna()
    if valid.sum() > 0:
        mean_abs = df.loc[valid, "abstractness"].mean()
        std_abs = df.loc[valid, "abstractness"].std()
        df["abstractness_z"] = np.nan
        df.loc[valid, "abstractness_z"] = (
            df.loc[valid, "abstractness"] - mean_abs
        ) / std_abs
    else:
        df["abstractness_z"] = np.nan
        mean_abs = std_abs = np.nan

    # Save
    output_path = args.output or args.scores
    df.to_csv(output_path, index=False)
    print(f"\nScored {n_scored} images, skipped {n_skipped}")
    if valid.sum() > 0:
        print(f"  Abstractness — mean: {mean_abs:.3f}, std: {std_abs:.3f}")
        print(
            f"  abstractness_z range: "
            f"[{df['abstractness_z'].min():.2f}, {df['abstractness_z'].max():.2f}]"
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
