#!/bin/bash
#SBATCH --job-name=nsd_setup_data
#SBATCH --output=slurm/slurm_outputs/setup_data_%j.out
#SBATCH --error=slurm/slurm_errors/setup_data_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Force unbuffered output so logs appear in real time ───────────────────────
export PYTHONUNBUFFERED=1

# ── Download all metadata (everything except betas) ───────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting setup_data --all (job $SLURM_JOB_ID)"

python -u scripts/setup_data.py --all

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished setup_data --all (job $SLURM_JOB_ID)"
