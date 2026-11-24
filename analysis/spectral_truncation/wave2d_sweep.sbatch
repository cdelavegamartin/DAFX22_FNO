#!/bin/bash
# 2D wave, original vs full-spectrum FourierConv2d (issue #2): one array task per line
# of analysis/spectral_truncation/wave2d_sweep_runs.txt, {original, full} x seeds 0-2.
# Every task trains FGRU, FRNN and REF together with joint clipping, as in the paper.
# Submit from the repository root, after `uv sync --extra cu126` and
# `uv run --extra cu126 python generate_dataset.py wave2d --steps 67`
# (output/datasets/wave2d-n_1024-steps_67-seed_0.npy, 66 output steps):
#   sbatch analysis/spectral_truncation/wave2d_sweep.sbatch           # all 6 runs
#   sbatch -a 1 analysis/spectral_truncation/wave2d_sweep.sbatch      # one line
# Arguments after the script name go to wave2d_sweep.py. A smoke test MUST write
# elsewhere, or --merge mixes it with the real runs; run one first to check GPU
# memory and the time per epoch:
#   sbatch -a 1 analysis/spectral_truncation/wave2d_sweep.sbatch --epochs 1 --out-dir output/smoke
#SBATCH -J dafx22-wave2d
#SBATCH -o %x.o%A_%a           # jobname.o<job id>_<array index>
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 8                   # 8 cores
#SBATCH --cpus-per-gpu=8
#SBATCH -t 96:0:0              # 96 hour runtime
#SBATCH --mem-per-cpu=7500M    # 8 * 7.5G = 60G total system RAM
#SBATCH --gres=gpu:1           # 1 GPU of any type
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="ampere|hopper"
#SBATCH -a 1-6

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" analysis/spectral_truncation/wave2d_sweep_runs.txt)
echo "task=$SLURM_ARRAY_TASK_ID arguments: $RUN host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086
python analysis/spectral_truncation/wave2d_sweep.py $RUN --device cuda "$@"
