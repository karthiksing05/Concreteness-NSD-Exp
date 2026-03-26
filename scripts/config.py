"""
config.py
=========
Central configuration for the NSD Visual Load & Abstractness experiment.
All paths, subject lists, ROI definitions, and S3 URLs in one place.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (auto-detected from this file's location)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# NSD local data root — downloaded NSD files go here
# ---------------------------------------------------------------------------
NSD_ROOT = DATA_DIR / "nsd"

# ---------------------------------------------------------------------------
# AWS S3 base URL (public, no credentials needed for HTTP downloads)
# ---------------------------------------------------------------------------
S3_BASE = "https://natural-scenes-dataset.s3.amazonaws.com"

# ---------------------------------------------------------------------------
# Subjects — NSD uses "subj01" through "subj08"
# ---------------------------------------------------------------------------
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
N_SUBJECTS = 8

# NSD experiment parameters
N_SESSIONS_MAX = 40
N_SESSIONS_PER_SUBJECT = 40  # max; actual varies per subject
TRIALS_PER_SESSION = 750
N_TOTAL_TRIALS = N_SESSIONS_PER_SUBJECT * TRIALS_PER_SESSION  # 30,000

# Actual sessions completed per subject (not all finished 40)
SESSIONS_PER_SUBJECT = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}

# ---------------------------------------------------------------------------
# Key NSD file paths (relative to NSD_ROOT)
# ---------------------------------------------------------------------------
NSD_EXPDESIGN_PATH = "nsddata/experiments/nsd/nsd_expdesign.mat"
NSD_STIM_INFO_PATH = "nsddata/experiments/nsd/nsd_stim_info_merged.csv"

# Betas (fsaverage surface space, assumehrf) — NSD uses .mgh for fsaverage
def beta_path(subject: str, hemi: str, session: int) -> str:
    """Return relative path for a single fsaverage beta file (.mgh)."""
    return (
        f"nsddata_betas/ppdata/{subject}/fsaverage/"
        f"betas_assumehrf/{hemi}.betas_session{session:02d}.mgh"
    )

# ROI masks (fsaverage surface space)
def roi_path(subject: str, hemi: str, roi_collection: str) -> str:
    """Return relative path for a surface ROI file."""
    return f"nsddata/freesurfer/{subject}/label/{hemi}.{roi_collection}.mgz"

# ROI masks (volume space, func1pt8mm)
def roi_volume_path(subject: str, roi_collection: str) -> str:
    """Return relative path for a volume ROI file."""
    return f"nsddata/ppdata/{subject}/func1pt8mm/roi/{roi_collection}.nii.gz"

# ROI label lookup files
def roi_ctab_path(subject: str, roi_collection: str) -> str:
    """Return relative path for ROI label-to-name mapping."""
    return f"nsddata/freesurfer/{subject}/label/{roi_collection}.mgz.ctab"

# ---------------------------------------------------------------------------
# ROI definitions
# ---------------------------------------------------------------------------
# Maps our analysis ROI names -> (roi_collection_file, integer_label)
# Based on NSD Data Manual:
#   prf-visualrois: 1=V1v, 2=V1d, 3=V2v, 4=V2d, 5=V3v, 6=V3d, 7=hV4
#   floc-places:    1=OPA, 2=PPA, 3=RSC
#   floc-faces:     1=OFA, 2=FFA-1, 3=FFA-2, 4=mTL-faces, 5=aTL-faces
#   floc-bodies:    1=EBA, 2=FBA-1, 3=FBA-2, 4=mTL-bodies
#   nsdgeneral:     binary (>0 = in ROI)
#   Kastner2015:    Wang et al. atlas with many regions

ROI_DEFINITIONS = {
    # Early visual cortex (from prf-visualrois)
    "V1v":  ("prf-visualrois", 1),
    "V1d":  ("prf-visualrois", 2),
    "V2v":  ("prf-visualrois", 3),
    "V2d":  ("prf-visualrois", 4),
    "V3v":  ("prf-visualrois", 5),
    "V3d":  ("prf-visualrois", 6),
    "hV4":  ("prf-visualrois", 7),
    # Combined early visual (union of V1v+V1d+V2v+V2d+V3v+V3d)
    "V1":   ("prf-visualrois", [1, 2]),   # V1v + V1d
    "V2":   ("prf-visualrois", [3, 4]),   # V2v + V2d
    "V3":   ("prf-visualrois", [5, 6]),   # V3v + V3d
    "EVC":  ("prf-visualrois", [1, 2, 3, 4, 5, 6]),  # V1+V2+V3 combined
    # Scene/place areas (from floc-places)
    "OPA":  ("floc-places", 1),
    "PPA":  ("floc-places", 2),
    "RSC":  ("floc-places", 3),
    # Face areas (from floc-faces)
    "OFA":  ("floc-faces", 1),
    "FFA1": ("floc-faces", 2),
    "FFA2": ("floc-faces", 3),
    # Body areas (from floc-bodies)
    "EBA":  ("floc-bodies", 1),
    # General visual cortex
    "nsdgeneral": ("nsdgeneral", None),  # binary mask, any value > 0
}

# The set of ROI collections we need to download
ROI_COLLECTIONS = sorted({v[0] for v in ROI_DEFINITIONS.values()})

# Default ROIs for our analysis pipeline
ANALYSIS_ROIS = [
    "nsdgeneral",
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4",
    "V1", "V2", "V3", "EVC",
    "OFA", "PPA", "RSC",
]

# ---------------------------------------------------------------------------
# COCO data
# ---------------------------------------------------------------------------
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_CAPTIONS_FILE = DATA_DIR / "coco" / "captions_train2017.json"
COCO_CAPTIONS_VAL_FILE = DATA_DIR / "coco" / "captions_val2017.json"
COCO_INSTANCES_FILE = DATA_DIR / "coco" / "instances_train2017.json"

# ---------------------------------------------------------------------------
# Brysbaert concreteness norms
# ---------------------------------------------------------------------------
BRYSBAERT_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.3758%2Fs13428-013-0403-5/MediaObjects/"
    "13428_2013_403_MOESM1_ESM.xlsx"
)
BRYSBAERT_FILE = DATA_DIR / "brysbaert_2014_concreteness.xlsx"

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
IMAGE_SCORES_FILE = DATA_DIR / "nsd_image_scores.csv"
SELECTED_IDS_FILE = DATA_DIR / "selected_image_ids.csv"
BETA_MAPS_DIR = RESULTS_DIR / "beta_maps"
RSA_RESULTS_DIR = RESULTS_DIR / "rsa_results"
GROUP_RESULTS_DIR = RESULTS_DIR / "group"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
FDR_Q = 0.05
MAX_COLLINEARITY = 0.3
N_PER_CELL = 30
SEED = 42
FD_THRESHOLD = 0.9  # mm, for motion exclusion
