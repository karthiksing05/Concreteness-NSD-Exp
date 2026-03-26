#!/bin/bash
#SBATCH --job-name=nsd_extract_betas
#SBATCH --output=slurm/slurm_outputs/extract_betas_%j.out
#SBATCH --error=slurm/slurm_errors/extract_betas_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_IDS="data/selected_image_ids.csv"
NSD_ROOT="data/nsd"
OUTPUT="results/beta_maps"

# ── Extract NSD betas for selected images ─────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting extract_betas (job $SLURM_JOB_ID)"

python scripts/extract_betas.py \
    --image_ids "$IMAGE_IDS" \
    --nsd_root "$NSD_ROOT" \
    --output "$OUTPUT" \
    --space fsaverage

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished extract_betas (job $SLURM_JOB_ID)"
