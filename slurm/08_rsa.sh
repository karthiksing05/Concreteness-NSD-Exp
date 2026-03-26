#!/bin/bash
#SBATCH --job-name=nsd_rsa
#SBATCH --output=slurm/slurm_outputs/rsa_%j.out
#SBATCH --error=slurm/slurm_errors/rsa_%j.err
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
NSD_ROOT="data/nsd"
OUTPUT="results/rsa_results"

# ── Representational Similarity Analysis ──────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RSA (job $SLURM_JOB_ID)"

python scripts/rsa.py \
    --betas "$BETAS" \
    --scores "$SCORES" \
    --output "$OUTPUT" \
    --nsd_root "$NSD_ROOT" \
    --rois nsdgeneral V1v V2v V3v hV4 OFA PPA RSC \
    --fdr_q 0.05

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished RSA (job $SLURM_JOB_ID)"
