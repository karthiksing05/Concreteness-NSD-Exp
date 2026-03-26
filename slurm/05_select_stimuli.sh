#!/bin/bash
#SBATCH --job-name=nsd_select_stimuli
#SBATCH --output=slurm/slurm_outputs/select_stimuli_%j.out
#SBATCH --error=slurm/slurm_errors/select_stimuli_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Paths ──────────────────────────────────────────────────────────────────────
SCORES="data/nsd_image_scores.csv"
OUTPUT="data/selected_image_ids.csv"

# ── Select balanced stimulus set (3x3 tertile grid) ──────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting select_stimuli (job $SLURM_JOB_ID)"

python scripts/select_stimuli.py \
    --scores "$SCORES" \
    --n_per_cell 30 \
    --output "$OUTPUT" \
    --max_collinearity 0.3 \
    --seed 42

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished select_stimuli (job $SLURM_JOB_ID)"
