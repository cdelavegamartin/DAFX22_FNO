#!/bin/bash
# Joint vs separate gradient clipping on the 1D strings: 4 array tasks,
# (string, nonlinear_string) x (joint, separate).
# Submit from the repository root, after `uv sync --extra cu126` and generating
# output/datasets/{string,nonlinear_string}-n_1024-seed_0.npy:
#   sbatch analysis/gradient_clipping/clip_test.sbatch
#SBATCH -J dafx22-clip
#SBATCH -o %x.o%A_%a           # jobname.o<job id>_<array index>
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 8                   # 8 cores
#SBATCH --cpus-per-gpu=8
#SBATCH -t 48:0:0              # 48 hour runtime
#SBATCH --mem-per-cpu=7500M    # 8 * 7.5G = 60G total system RAM
#SBATCH --gres=gpu:1           # 1 GPU of any type
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="ampere|hopper"
#SBATCH -a 0-3

set -euo pipefail

SYSTEMS=(string string nonlinear_string nonlinear_string)
CONDITIONS=(joint separate joint separate)
SYSTEM=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "system=$SYSTEM condition=$CONDITION host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python analysis/gradient_clipping/clip_test_1d.py \
    --system "$SYSTEM" --condition "$CONDITION" --device cuda "$@"
