#!/bin/bash
#SBATCH --job-name=nsd_setup_betas
#SBATCH --output=slurm/slurm_outputs/setup_betas_%j.out
#SBATCH --error=slurm/slurm_errors/setup_betas_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Download NSD betas for all subjects (LARGE download) ──────────────────────
# Adjust --subjects to download a subset, e.g. --subjects subj01 subj02
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting setup_data --betas (job $SLURM_JOB_ID)"

python scripts/setup_data.py \
    --betas \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished setup_data --betas (job $SLURM_JOB_ID)"
