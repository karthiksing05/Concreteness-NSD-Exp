#!/bin/bash
#SBATCH --job-name=nsd_score_load
#SBATCH --output=slurm/slurm_outputs/score_load_%j.out
#SBATCH --error=slurm/slurm_errors/score_load_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/Concreteness-NSD-Exp

# ── Paths ──────────────────────────────────────────────────────────────────────
NSD_IMAGE_DIR="data/nsd/nsddata_stimuli/stimuli/nsd"
COCO_INSTANCES="data/coco/instances_train2017.json"
NSD_STIM_INFO="data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
OUTPUT="data/nsd_image_scores.csv"

# ── Compute visual load (entropy) scores for all NSD images ───────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting score_load (job $SLURM_JOB_ID)"

python scripts/score_load.py \
    --nsd_image_dir "$NSD_IMAGE_DIR" \
    --output "$OUTPUT" \
    --coco_instances "$COCO_INSTANCES" \
    --nsd_stim_info "$NSD_STIM_INFO" \
    --workers 8

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished score_load (job $SLURM_JOB_ID)"
