#!/bin/bash
#SBATCH --job-name=nsd_score_abstract
#SBATCH --output=slurm/slurm_outputs/score_abstract_%j.out
#SBATCH --error=slurm/slurm_errors/score_abstract_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Paths ──────────────────────────────────────────────────────────────────────
SCORES="data/nsd_image_scores.csv"
BRYSBAERT="data/brysbaert_2014_concreteness.xlsx"
COCO_CAPTIONS="data/coco/captions_train2017.json"
NSD_STIM_INFO="data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv"

# ── Ensure NLTK data is available ─────────────────────────────────────────────
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# ── Compute abstractness scores ───────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting score_abstractness (job $SLURM_JOB_ID)"

python scripts/score_abstractness.py \
    --scores "$SCORES" \
    --brysbaert "$BRYSBAERT" \
    --coco_captions "$COCO_CAPTIONS" \
    --nsd_stim_info "$NSD_STIM_INFO"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished score_abstractness (job $SLURM_JOB_ID)"
