#!/usr/bin/env python3
"""
setup_data.py
=============
Download and organize all external data needed for the NSD experiment.

Downloads:
  1. NSD experiment design (nsd_expdesign.mat)           ~small
  2. NSD stimulus info (nsd_stim_info_merged)             ~small
  3. NSD ROI masks (fsaverage surface) for all subjects   ~small per subject
  4. NSD betas for selected subjects (LARGE — selective)
  5. COCO annotations (captions + instances)              ~241MB zip
  6. Brysbaert concreteness norms                         ~2MB

Usage:
    # Download everything except betas (metadata, masks, annotations)
    python scripts/setup_data.py --all

    # Download only specific components
    python scripts/setup_data.py --expdesign --rois --coco --brysbaert

    # Download betas for specific subjects (LARGE)
    python scripts/setup_data.py --betas --subjects subj01 subj02

    # Download betas for specific sessions only
    python scripts/setup_data.py --betas --subjects subj01 --sessions 1 2 3

    # Check what's already downloaded
    python scripts/setup_data.py --status
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# Add scripts dir to path for config import
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR,
    NSD_ROOT,
    S3_BASE,
    SUBJECTS,
    N_SESSIONS_PER_SUBJECT,
    SESSIONS_PER_SUBJECT,
    NSD_EXPDESIGN_PATH,
    NSD_STIM_INFO_PATH,
    ROI_COLLECTIONS,
    COCO_ANNOTATIONS_URL,
    COCO_CAPTIONS_FILE,
    BRYSBAERT_URL,
    BRYSBAERT_FILE,
    beta_path,
    roi_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file from URL to dest. Returns True on success."""
    dest = Path(dest)
    if dest.exists():
        print(f"  [SKIP] {desc or dest.name} — already exists")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [GET]  {desc or dest.name}")
    print(f"         {url}")

    try:
        tmp_path = str(dest) + ".tmp"
        urlretrieve(url, tmp_path)
        os.rename(tmp_path, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"         Done ({size_mb:.1f} MB)")
        return True
    except (HTTPError, URLError, OSError) as e:
        print(f"  [FAIL] {desc or dest.name}: {e}")
        # Clean up partial download
        tmp_path = str(dest) + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def nsd_url(relative_path: str) -> str:
    """Build full S3 URL from NSD-relative path."""
    return f"{S3_BASE}/{relative_path}"


def nsd_local(relative_path: str) -> Path:
    """Build local path from NSD-relative path."""
    return NSD_ROOT / relative_path


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_expdesign() -> bool:
    """Download NSD experiment design file."""
    print("\n=== NSD Experiment Design ===")
    return download_file(
        nsd_url(NSD_EXPDESIGN_PATH),
        nsd_local(NSD_EXPDESIGN_PATH),
        "nsd_expdesign.mat",
    )


def download_stim_info() -> bool:
    """Download NSD stimulus info (nsd_id -> coco_id mapping)."""
    print("\n=== NSD Stimulus Info ===")

    # Try multiple possible filenames
    candidates = [
        "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
        "nsddata/experiments/nsd/nsd_stim_info_merged.pkl",
    ]

    for relpath in candidates:
        ok = download_file(
            nsd_url(relpath), nsd_local(relpath), Path(relpath).name
        )
        if ok:
            return True

    # Also try downloading the COCO-73k ordering file
    coco73k_path = "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    print(f"  [INFO] Stim info may need manual download or alternative access.")
    print(f"         Check: {S3_BASE}/nsddata/experiments/nsd/")
    return False


def download_rois(subjects: list[str] | None = None) -> bool:
    """Download ROI mask files for all subjects."""
    print("\n=== NSD ROI Masks (fsaverage surface) ===")
    subjects = subjects or SUBJECTS
    all_ok = True

    for subject in subjects:
        print(f"\n  Subject: {subject}")
        for collection in ROI_COLLECTIONS:
            for hemi in ["lh", "rh"]:
                relpath = roi_path(subject, hemi, collection)
                ok = download_file(
                    nsd_url(relpath), nsd_local(relpath),
                    f"  {hemi}.{collection}.mgz"
                )
                if not ok:
                    all_ok = False

            # Also download the ctab label file
            ctab_relpath = (
                f"nsddata/freesurfer/{subject}/label/{collection}.mgz.ctab"
            )
            download_file(
                nsd_url(ctab_relpath), nsd_local(ctab_relpath),
                f"  {collection}.mgz.ctab"
            )

    # Also download fsaverage probabilistic maps
    print(f"\n  Probabilistic maps (fsaverage):")
    prob_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4",
                 "OPA", "PPA", "RSC", "OFA", "FFA-1", "FFA-2", "EBA"]
    for roi in prob_rois:
        for hemi in ["lh", "rh"]:
            relpath = f"nsddata/freesurfer/fsaverage/label/{hemi}.{roi}.mgz"
            download_file(
                nsd_url(relpath), nsd_local(relpath),
                f"  fsaverage/{hemi}.{roi}.mgz"
            )

    return all_ok


def download_betas(
    subjects: list[str] | None = None,
    sessions: list[int] | None = None,
) -> bool:
    """
    Download NSD beta files (fsaverage, betas_assumehrf).
    WARNING: Each session file is ~500MB-1GB. Full download is ~8TB.
    """
    print("\n=== NSD Betas (fsaverage, betas_assumehrf) ===")
    subjects = subjects or SUBJECTS

    for subject in subjects:
        n_sessions = SESSIONS_PER_SUBJECT.get(subject, N_SESSIONS_PER_SUBJECT)
        sub_sessions = sessions or list(range(1, n_sessions + 1))
        total_files = len(sub_sessions) * 2  # 2 hemispheres
        print(f"  {subject}: {n_sessions} sessions, {total_files} files")

    print("  WARNING: Each .mgh file is ~250-500MB. This will take significant time/space.")

    response = input("  Continue? [y/N]: ").strip().lower()
    if response != "y":
        print("  Skipping beta download.")
        return False

    all_ok = True
    for subject in subjects:
        n_sessions = SESSIONS_PER_SUBJECT.get(subject, N_SESSIONS_PER_SUBJECT)
        sub_sessions = sessions or list(range(1, n_sessions + 1))
        print(f"\n  Subject: {subject} ({len(sub_sessions)} sessions)")
        for session in sub_sessions:
            for hemi in ["lh", "rh"]:
                relpath = beta_path(subject, hemi, session)
                ok = download_file(
                    nsd_url(relpath), nsd_local(relpath),
                    f"  {hemi}.betas_session{session:02d}.nii.gz"
                )
                if not ok:
                    all_ok = False
    return all_ok


def download_coco() -> bool:
    """Download COCO annotations (captions and instances)."""
    print("\n=== COCO Annotations ===")

    coco_dir = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if COCO_CAPTIONS_FILE.exists():
        print(f"  [SKIP] COCO captions already exist at {COCO_CAPTIONS_FILE}")
        return True

    # Download the zip
    zip_path = coco_dir / "annotations_trainval2017.zip"
    ok = download_file(COCO_ANNOTATIONS_URL, zip_path, "annotations_trainval2017.zip")
    if not ok:
        return False

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Extract to coco dir, flattening the 'annotations/' prefix
            basename = os.path.basename(member)
            if basename and basename.endswith(".json"):
                target = coco_dir / basename
                if not target.exists():
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"    Extracted: {basename}")

    # Clean up zip
    zip_path.unlink()
    print("  Removed zip file")
    return True


def download_brysbaert() -> bool:
    """Download Brysbaert et al. (2014) concreteness norms."""
    print("\n=== Brysbaert Concreteness Norms ===")
    ok = download_file(BRYSBAERT_URL, BRYSBAERT_FILE, "brysbaert_2014_concreteness.xlsx")
    if not ok:
        print("  [INFO] If automatic download fails, manually download from:")
        print("         https://doi.org/10.3758/s13428-013-0403-5")
        print("         (Supplementary Material → ESM 1)")
        print(f"         Save as: {BRYSBAERT_FILE}")
    return ok


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def check_status():
    """Print status of all required data files."""
    print("\n" + "=" * 60)
    print("DATA STATUS")
    print("=" * 60)

    def check(path: Path, label: str):
        exists = path.exists()
        size = ""
        if exists:
            s = path.stat().st_size
            if s > 1024 * 1024:
                size = f" ({s / (1024*1024):.1f} MB)"
            elif s > 1024:
                size = f" ({s / 1024:.1f} KB)"
        status = "OK" if exists else "MISSING"
        icon = "[x]" if exists else "[ ]"
        print(f"  {icon} {label}{size}")

    print("\n  --- Core metadata ---")
    check(nsd_local(NSD_EXPDESIGN_PATH), "NSD experiment design")
    check(nsd_local(NSD_STIM_INFO_PATH), "NSD stim info")

    print("\n  --- Annotations ---")
    check(COCO_CAPTIONS_FILE, "COCO captions (train)")
    check(DATA_DIR / "coco" / "captions_val2017.json", "COCO captions (val)")
    check(DATA_DIR / "coco" / "instances_train2017.json", "COCO instances (train)")
    check(BRYSBAERT_FILE, "Brysbaert norms")

    print("\n  --- ROI masks ---")
    for subject in SUBJECTS:
        n_found = 0
        n_total = 0
        for collection in ROI_COLLECTIONS:
            for hemi in ["lh", "rh"]:
                n_total += 1
                if nsd_local(roi_path(subject, hemi, collection)).exists():
                    n_found += 1
        status = "OK" if n_found == n_total else f"{n_found}/{n_total}"
        icon = "[x]" if n_found == n_total else "[ ]"
        print(f"  {icon} {subject} ROIs: {status}")

    print("\n  --- Betas (fsaverage) ---")
    for subject in SUBJECTS:
        n_sessions = SESSIONS_PER_SUBJECT.get(subject, N_SESSIONS_PER_SUBJECT)
        n_found = 0
        n_total = n_sessions * 2
        for session in range(1, n_sessions + 1):
            for hemi in ["lh", "rh"]:
                if nsd_local(beta_path(subject, hemi, session)).exists():
                    n_found += 1
        if n_found == 0:
            print(f"  [ ] {subject} betas: none downloaded")
        elif n_found == n_total:
            print(f"  [x] {subject} betas: all {n_total} files")
        else:
            print(f"  [~] {subject} betas: {n_found}/{n_total} files")

    print("\n  --- Processed data ---")
    check(DATA_DIR / "nsd_image_scores.csv", "Image scores")
    check(DATA_DIR / "selected_image_ids.csv", "Selected image IDs")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and set up all data for the NSD experiment."
    )
    parser.add_argument("--all", action="store_true",
                        help="Download everything except betas")
    parser.add_argument("--expdesign", action="store_true",
                        help="Download NSD experiment design")
    parser.add_argument("--stim-info", action="store_true",
                        help="Download NSD stimulus info")
    parser.add_argument("--rois", action="store_true",
                        help="Download ROI masks for all subjects")
    parser.add_argument("--betas", action="store_true",
                        help="Download beta files (LARGE)")
    parser.add_argument("--coco", action="store_true",
                        help="Download COCO annotations")
    parser.add_argument("--brysbaert", action="store_true",
                        help="Download Brysbaert concreteness norms")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subjects for ROI/beta download (default: all)")
    parser.add_argument("--sessions", nargs="+", type=int, default=None,
                        help="Sessions for beta download (default: all)")
    parser.add_argument("--status", action="store_true",
                        help="Check download status")
    args = parser.parse_args()

    # If no flags, show help
    if not any([args.all, args.expdesign, args.stim_info, args.rois,
                args.betas, args.coco, args.brysbaert, args.status]):
        parser.print_help()
        print("\nTip: Run with --status to see what's already downloaded.")
        return

    if args.status:
        check_status()
        return

    print("NSD Experiment — Data Setup")
    print(f"Data directory: {DATA_DIR}")
    print(f"NSD root: {NSD_ROOT}")

    results = {}

    if args.all or args.expdesign:
        results["expdesign"] = download_expdesign()

    if args.all or args.stim_info:
        results["stim_info"] = download_stim_info()

    if args.all or args.rois:
        results["rois"] = download_rois(args.subjects)

    if args.betas:
        results["betas"] = download_betas(args.subjects, args.sessions)

    if args.all or args.coco:
        results["coco"] = download_coco()

    if args.all or args.brysbaert:
        results["brysbaert"] = download_brysbaert()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        icon = "[x]" if ok else "[!]"
        print(f"  {icon} {name}")

    if all(results.values()):
        print("\nAll downloads successful!")
    else:
        print("\nSome downloads failed. Run with --status to check.")

    check_status()


if __name__ == "__main__":
    main()
