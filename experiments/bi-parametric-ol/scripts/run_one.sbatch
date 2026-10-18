#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -euo pipefail
export OMP_NUM_THREADS=4
nvidia-smi -L
python "$SLURM_SUBMIT_DIR/run.py" "$RUN" ${OUT:+--out "$OUT"}
