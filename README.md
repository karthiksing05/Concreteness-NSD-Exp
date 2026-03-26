# NSD Visual Load & Abstractness: Neural Activation Study

A neuroscience experiment using the Natural Scenes Dataset (NSD) to investigate how the human brain differentially processes images that vary along two axes: **visual stimulation load** (low → high) and **conceptual abstractness** (concrete → abstract). Brain activations are compared across these axes using voxelwise regression and Representational Similarity Analysis (RSA).

---

## Table of Contents

1. [Overview](#overview)
2. [Hypotheses](#hypotheses)
3. [Dataset](#dataset)
4. [Image Scoring Pipeline](#image-scoring-pipeline)
   - [Stimulation Load Score](#stimulation-load-score)
   - [Abstractness Score](#abstractness-score)
5. [Stimulus Selection](#stimulus-selection)
6. [fMRI Data Extraction](#fmri-data-extraction)
7. [ROI Definition](#roi-definition)
8. [Analysis Pipeline](#analysis-pipeline)
   - [Voxelwise Regression](#voxelwise-regression)
   - [RSA](#representational-similarity-analysis-rsa)
9. [Confound Control](#confound-control)
10. [Directory Structure](#directory-structure)
11. [Dependencies](#dependencies)
12. [Usage](#usage)

---

## Overview

The Natural Scenes Dataset (NSD) provides single-trial fMRI beta estimates for ~73,000 images shown to 8 subjects. Rather than treating stimulation load and abstractness as discrete categories, this study scores every NSD image along both dimensions continuously, then uses those scores as regressors against BOLD responses across the brain.

The goal is to produce:
- A voxelwise map of sensitivity to **visual load** across the cortex
- A voxelwise map of sensitivity to **conceptual abstractness**
- A test of the **interaction**: does visual load modulate how abstract vs. concrete content is processed in higher cortical areas?

---

## Hypotheses

| # | Hypothesis |
|---|-----------|
| H1 | Stimulation load (high > low) drives stronger activation in early visual cortex (V1–V4) and plateaus or inverts in higher areas due to normalization/adaptation |
| H2 | Abstractness (abstract > concrete) preferentially engages default mode regions (dmPFC, TPJ, angular gyrus) and left inferior frontal gyrus |
| H3 | A load × abstractness interaction exists in lateral occipital cortex (LOC) and scene-selective regions (PPA, RSC), reflecting modulation of object/scene coding by visual complexity |

---

## Dataset

**Natural Scenes Dataset (NSD)**
- 8 subjects (sub-01 through sub-08), each seeing up to 30,000 unique COCO images during fMRI
- Pre-computed single-trial beta estimates (`betas_assumehrf`) in both fsaverage surface and MNI volume space
- COCO image annotations (category labels, captions, segmentation masks) included
- Access: https://naturalscenesdataset.org

**Brysbaert et al. (2014) Concreteness Norms**
- 37,058 English words rated on a 1–5 concreteness scale (1 = abstract, 5 = concrete)
- Used to score image abstractness via COCO captions
- Access: https://doi.org/10.3758/s13428-013-0403-5 (supplementary data)

---

## Image Scoring Pipeline

All scoring is done on the 73,000 NSD/COCO images prior to any fMRI data extraction. Each image receives two scalar scores.

### Stimulation Load Score

Visual stimulation load is operationalized as **pixel-level entropy** — the Shannon entropy of the image's luminance histogram. This is simple, interpretable, fast to compute, and well-validated as a proxy for visual complexity.

```
H = - Σ p(i) * log2(p(i))
```

where `p(i)` is the probability of luminance bin `i` in the grayscale histogram.

**Why entropy?** It captures both low-level contrast spread and high-level scene clutter in a single number. A blank grey field has near-zero entropy; a crowded market scene approaches the maximum. It avoids the need for object detection or saliency models, keeping the pipeline self-contained.

**Supplementary metrics (optional validation):**
- **Edge density**: mean Canny edge magnitude (captures spatial frequency content)
- **Object count**: from COCO instance annotations (integer count per image)

These can be used to validate the entropy scores and as additional regressors in a robustness analysis, but entropy is the primary operationalization.

**Implementation** (`scripts/score_load.py`):

```python
import numpy as np
from PIL import Image
from scipy.stats import entropy as scipy_entropy

def image_entropy(img_path: str) -> float:
    """Compute Shannon entropy of the luminance histogram."""
    img = Image.open(img_path).convert("L")
    hist, _ = np.histogram(np.array(img).ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # normalize to probability distribution
    return float(scipy_entropy(hist, base=2))
```

Scores are z-scored across the full NSD corpus before use as regressors.

---

### Abstractness Score

Each image is scored using the **mean concreteness** of content words in its COCO caption(s), mapped through the Brysbaert et al. (2014) norms. Abstractness is then the inverse: `abstractness = 5.0 - mean_concreteness`.

**Steps:**

1. Load all COCO captions for each image (typically 5 captions per image in COCO)
2. Tokenize and lowercase; remove stopwords and punctuation
3. For each remaining word, look up its concreteness rating in the Brysbaert norms
4. Average across all matched words across all captions for that image
5. Invert: `abstractness = 5.0 - mean_concreteness`
6. Images with fewer than 3 matched words are excluded

**Implementation** (`scripts/score_abstractness.py`):

```python
import json
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def load_brysbaert(path: str) -> dict:
    """Returns {word: concreteness_rating} dict."""
    df = pd.read_excel(path)
    return dict(zip(df["Word"].str.lower(), df["Conc.M"]))

def score_image_abstractness(captions: list[str], norms: dict) -> float | None:
    stops = set(stopwords.words("english"))
    ratings = []
    for cap in captions:
        tokens = [w.lower() for w in word_tokenize(cap)
                  if w.isalpha() and w.lower() not in stops]
        ratings.extend([norms[t] for t in tokens if t in norms])
    if len(ratings) < 3:
        return None
    mean_concreteness = np.mean(ratings)
    return float(5.0 - mean_concreteness)  # invert: high = abstract
```

Scores are z-scored across the full NSD corpus before use as regressors.

---

## Stimulus Selection

From the scored corpus, select ~200–300 images that:

- Have valid scores on both axes (not excluded from abstractness scoring)
- Span the full range of both dimensions (avoid clustering in one corner of the space)
- Have been seen by **all 8 subjects** in NSD (use the shared 1,000-image NSD core set as a starting point — all 8 subjects saw these)

**Sampling strategy**: Divide each axis into tertiles (low / mid / high) and sample ~25–35 images uniformly from each of the 9 resulting cells. This ensures coverage without hard categorization and allows continuous analysis.

Verify orthogonality: compute Pearson r between load and abstractness scores for the selected set. Aim for |r| < 0.3. If collinear, resample.

---

## fMRI Data Extraction

NSD provides pre-computed GLM betas via the NSD data server. No re-fitting of the GLM is required.

**Data files used:**
- `betas_assumehrf` — single-trial beta estimates assuming canonical HRF
- Format: fsaverage surface (recommended) or MNI volume
- Resolution: 1.8mm isotropic

**Extraction steps:**

1. For each subject, load the beta file corresponding to each selected image's trial index
2. Map trial indices using the NSD `expdesign.mat` file which records which image was shown on which trial
3. For images seen multiple times by the same subject (NSD shows each shared image up to 3×), average the betas across repetitions — this increases SNR substantially
4. Apply ROI mask (see below) to extract region-specific response vectors

```python
# Pseudocode — adapt to your NSD data access method
import numpy as np

def extract_betas(subject_id, trial_indices, roi_mask, nsd_data_root):
    betas = []
    for trial_idx in trial_indices:
        b = load_nsd_beta(subject_id, trial_idx, nsd_data_root)  # shape: (n_voxels,)
        betas.append(b[roi_mask])
    return np.stack(betas)  # shape: (n_images, n_voxels_in_roi)
```

---

## ROI Definition

Use the NSD-provided parcellation (`nsdgeneral` and `streams` atlases in fsaverage space).

| ROI | Label | Role in this study |
|-----|-------|--------------------|
| V1–V4 | Early visual | Load sensitivity (low-level) |
| LOC | Lateral occipital complex | Object selectivity × load |
| OFA | Occipital face area | Object-level coding |
| PPA | Parahippocampal place area | Scene/context selectivity |
| RSC | Retrosplenial complex | Scene/spatial coding |
| IPS | Intraparietal sulcus | Attentional load |
| dmPFC | Dorsomedial prefrontal | Abstract concept processing |
| TPJ | Temporoparietal junction | Abstract/social cognition |
| AngG | Angular gyrus | Semantic/abstract language |

ROI masks are provided in NSD's `freesurfer/fsaverage/label/` directory. For volume-space analyses, use the NSD MNI-space parcellation.

---

## Analysis Pipeline

### Voxelwise Regression

For each voxel `v` in each subject, fit:

```
BOLD_v ~ β1·load + β2·abstract + β3·(load × abstract) + β4·mean_lum + β5·rms_contrast + ε
```

Where:
- `load` = z-scored entropy score
- `abstract` = z-scored abstractness score
- `load × abstract` = element-wise product of the two z-scored regressors
- `mean_lum`, `rms_contrast` = nuisance regressors for low-level image properties

**Output:** Maps of β1, β2, β3 across the cortical surface for each subject. Group-level maps are computed by averaging subject-level β maps and testing against zero with a one-sample t-test, FDR corrected at q < 0.05.

**Implementation** (`scripts/voxelwise_regression.py`):

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def fit_voxelwise(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    X: (n_images, n_regressors) design matrix
    Y: (n_images, n_voxels) BOLD responses
    Returns: (n_regressors, n_voxels) beta map
    """
    reg = LinearRegression(fit_intercept=True).fit(X, Y)
    return reg.coef_.T  # (n_regressors, n_voxels)
```

---

### Representational Similarity Analysis (RSA)

RSA asks whether the *geometry* of neural representations in a region mirrors the geometry of the stimulus space.

**Steps:**

1. **Model RDM**: Compute the pairwise Euclidean distance matrix between all selected images in (load, abstract) score space → matrix of shape (n_images, n_images)
2. **Neural RDM**: For each ROI, compute pairwise 1 − Pearson correlation between BOLD response patterns → (n_images, n_images) per ROI per subject
3. **RSA correlation**: Spearman correlation between the upper triangles of the model RDM and each neural RDM
4. Group-level: one-sample t-test on Fisher-z-transformed RSA correlations across subjects, FDR corrected

You can also test model RDMs for each axis independently (load-only RDM, abstract-only RDM) to disentangle which regions preferentially encode each dimension.

```python
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import numpy as np

def compute_rdm_neural(betas: np.ndarray) -> np.ndarray:
    """betas: (n_images, n_voxels) → RDM: (n_images, n_images)"""
    corr = np.corrcoef(betas)
    return 1 - corr

def rsa_correlation(model_rdm: np.ndarray, neural_rdm: np.ndarray) -> float:
    model_vec = squareform(model_rdm)
    neural_vec = squareform(neural_rdm)
    r, _ = spearmanr(model_vec, neural_vec)
    return float(r)
```

---

## Confound Control

| Confound | Measure | Treatment |
|----------|---------|-----------|
| Mean luminance | Mean pixel intensity | Nuisance regressor in GLM |
| RMS contrast | Std of pixel intensities | Nuisance regressor in GLM |
| Color saturation | Mean HSV saturation | Nuisance regressor if using color images |
| Load ↔ abstractness collinearity | Pearson r between scores | Verify |r| < 0.3 in selected set; report VIF |
| Subject-level head motion | Framewise displacement from NSD | Exclude trials with FD > 0.9mm |

---

## Directory Structure

```
nsd-load-abstract/
├── README.md
├── data/
│   ├── brysbaert_2014_concreteness.xlsx   # Brysbaert norms (download separately)
│   ├── nsd_image_scores.csv               # Output of scoring pipeline
│   └── selected_image_ids.csv             # Final stimulus set
├── scripts/
│   ├── score_load.py                      # Entropy-based load scoring
│   ├── score_abstractness.py              # Brysbaert-based abstractness scoring
│   ├── select_stimuli.py                  # Balanced sampling from score space
│   ├── extract_betas.py                   # NSD beta extraction
│   ├── voxelwise_regression.py            # GLM across voxels
│   └── rsa.py                             # RSA pipeline
├── results/
│   ├── beta_maps/                         # Per-subject voxelwise β maps
│   ├── rsa_results/                       # RSA r-values per ROI per subject
│   └── group/                             # Group-level t-maps and figures
└── notebooks/
    ├── 01_score_images.ipynb
    ├── 02_check_orthogonality.ipynb
    ├── 03_extract_and_regress.ipynb
    └── 04_rsa_and_figures.ipynb
```

---

## Dependencies

```
python >= 3.10
numpy
scipy
pandas
Pillow
nltk
scikit-learn
nibabel          # NIfTI volume loading
nilearn          # ROI masking, surface projection
matplotlib
seaborn
openpyxl         # for reading Brysbaert .xlsx
```

Install:
```bash
pip install numpy scipy pandas Pillow nltk scikit-learn nibabel nilearn matplotlib seaborn openpyxl
python -m nltk.downloader stopwords punkt
```

NSD data access requires registration at https://naturalscenesdataset.org.

---

## Usage

```bash
# 1. Score all NSD images on both axes
python scripts/score_load.py --nsd_image_dir /path/to/nsd/images --output data/nsd_image_scores.csv

# 2. Add abstractness scores
python scripts/score_abstractness.py --scores data/nsd_image_scores.csv \
    --brysbaert data/brysbaert_2014_concreteness.xlsx \
    --coco_captions /path/to/coco/annotations/captions_train2017.json

# 3. Select stimulus set
python scripts/select_stimuli.py --scores data/nsd_image_scores.csv \
    --n_per_cell 30 --output data/selected_image_ids.csv

# 4. Extract NSD betas for selected images
python scripts/extract_betas.py --image_ids data/selected_image_ids.csv \
    --nsd_root /path/to/nsd --output results/beta_maps/

# 5. Run voxelwise regression
python scripts/voxelwise_regression.py --betas results/beta_maps/ \
    --scores data/nsd_image_scores.csv --output results/

# 6. Run RSA
python scripts/rsa.py --betas results/beta_maps/ \
    --scores data/nsd_image_scores.csv --output results/rsa_results/
```

---

## References

- Allen, E.J. et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25, 116–126.
- Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings for 37,058 English words and two-character Chinese words. *Behavior Research Methods*, 46, 904–911.
- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis — connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.