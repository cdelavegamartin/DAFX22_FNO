# Joint vs separate gradient clipping (1D string)

`train_1d_string.py` (commit `e08b4b0`, Parker et al. 2022) trains FGRU, FRNN and REF with one AdamW over all their parameters and clips the norm of their concatenated gradients to 1.
The loss is log10(MSE), whose gradient is grad(MSE) / (MSE ln 10), so the model with the lowest loss tends to dominate the joint norm, and the clip then shrinks the steps of the others.
This may explain why REF performs poorly on the 1D string in the paper (validation MSE 0.350 vs 7.79e-3 for FGRU).

`clip_test_1d.py` trains the three models under two conditions that differ only in clipping:

- `joint`: one AdamW and one OneCycleLR over all parameters, joint clipping (the paper).
- `separate`: one AdamW and one OneCycleLR per model, each model clipped on its own.

Both conditions use the paper's data (1024 trajectories, half plucks, half random; 119 output steps on 65 points), widths, learning rates and batch size, the same initial weights and the same batch order.
AdamW and OneCycleLR act per parameter, so clipping is the only difference.
Per epoch, the script logs each model's training MSE, its gradient norm before clipping, and the factor by which clipping scales its gradient.

## Run

`--system` selects the linear (`string`, width 16) or the tension-modulated string (`nonlinear_string`, width 32); the other settings are the same for both, as in the paper.
Each condition needs about 8 GB of memory (autograd over 119 steps at batch 512); the paper's 5000 epochs need a GPU.

```bash
uv sync --extra cu126
uv run python generate_dataset.py string             # output/datasets/string-n_1024-seed_0.npy
uv run python generate_dataset.py nonlinear_string   # output/datasets/nonlinear_string-n_1024-seed_0.npy
uv run python analysis/gradient_clipping/clip_test_1d.py --system string --condition joint --device cuda
uv run python analysis/gradient_clipping/clip_test_1d.py --system string --condition separate --device cuda
uv run python analysis/gradient_clipping/clip_test_1d.py --system string --merge
```

Each run writes `results/<system>/<condition>.json`; `--merge` writes `results/<system>/results.json` (validation MSE per model, per-step validation MSE, training logs) and `results/<system>/clipping.png` (training MSE, gradient norms and clipping scale per model and condition).
`--data` and `--out-dir` override the default paths.

### SLURM

`clip_test.sbatch` runs the 4 combinations (string, nonlinear_string) x (joint, separate) as array tasks 0–3, one GPU each.
From the repository root, after `uv sync --extra cu126` and generating both datasets:

```bash
sbatch analysis/gradient_clipping/clip_test.sbatch
# afterwards, on any node:
uv run python analysis/gradient_clipping/clip_test_1d.py --system string --merge
uv run python analysis/gradient_clipping/clip_test_1d.py --system nonlinear_string --merge
```

Arguments after the script name go to `clip_test_1d.py`, e.g. a one-epoch check: `sbatch analysis/gradient_clipping/clip_test.sbatch --epochs 1`.
Account, partition, runtime and resources are in the `#SBATCH` lines.

## Expected signature

If joint clipping explains the poor REF results, under `joint` REF's clipping scale stays well below 1 while FGRU's gradient norm dominates the joint norm, and REF's training and validation MSE improve under `separate`.
