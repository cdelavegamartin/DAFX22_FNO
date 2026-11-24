#!/bin/bash
# Clipping sweep on the 1D strings: one array task per line of
# analysis/gradient_clipping/sweep_runs.txt (44 lines, made by make_sweep_runs.py):
# 2 systems x seeds 0-2 x max norm {1, 10, 100, inf} x {joint, separate}, minus the
# 4 runs already in results/ (seed 0, max norm 1).
# Submit from the repository root, after `uv sync --extra cu126` and generating
# output/datasets/{string,nonlinear_string}-n_1024-seed_0.npy:
#   sbatch analysis/gradient_clipping/clip_sweep.sbatch               # all runs
#   sbatch -a 1-4 analysis/gradient_clipping/clip_sweep.sbatch        # a subset of lines
#   sbatch -a 1-44%20 analysis/gradient_clipping/clip_sweep.sbatch    # at most 20 at a time
# Arguments after the script name go to clip_test_1d.py. A smoke test MUST write
# elsewhere, or --merge mixes it with the real runs:
#   sbatch -a 1 analysis/gradient_clipping/clip_sweep.sbatch --epochs 1 --out-dir output/smoke
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
#SBATCH -a 1-44

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" analysis/gradient_clipping/sweep_runs.txt)
echo "task=$SLURM_ARRAY_TASK_ID arguments: $RUN host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086
python analysis/gradient_clipping/clip_test_1d.py $RUN --device cuda "$@"
