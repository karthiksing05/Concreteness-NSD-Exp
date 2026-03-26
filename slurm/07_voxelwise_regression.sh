#!/bin/bash
#SBATCH --job-name=nsd_voxelwise_glm
#SBATCH --output=slurm/slurm_outputs/voxelwise_regression_%j.out
#SBATCH --error=slurm/slurm_errors/voxelwise_regression_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Paths ──────────────────────────────────────────────────────────────────────
BETAS="results/beta_maps"
SCORES="data/nsd_image_scores.csv"
OUTPUT="results"

# ── Voxelwise regression: BOLD ~ load + abstract + interaction + nuisance ────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting voxelwise_regression (job $SLURM_JOB_ID)"

python scripts/voxelwise_regression.py \
    --betas "$BETAS" \
    --scores "$SCORES" \
    --output "$OUTPUT" \
    --fdr_q 0.05

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished voxelwise_regression (job $SLURM_JOB_ID)"
