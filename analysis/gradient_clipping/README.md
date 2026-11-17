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
`--seed` sets the initial weights and the batch order, and `--max-norm` the clipping threshold (`inf` disables clipping); the dataset stays the seed-0 one.
Each condition needs about 8 GB of memory (autograd over 119 steps at batch 512); the paper's 5000 epochs need a GPU.

```bash
uv sync --extra cu126
uv run --extra cu126 python generate_dataset.py string             # output/datasets/string-n_1024-seed_0.npy
uv run --extra cu126 python generate_dataset.py nonlinear_string   # output/datasets/nonlinear_string-n_1024-seed_0.npy
uv run --extra cu126 python analysis/gradient_clipping/clip_test_1d.py --system string --condition joint --seed 1 --max-norm 10 --device cuda
uv run --extra cu126 python analysis/gradient_clipping/clip_test_1d.py --merge
```

`uv run` needs the backend extra (`--extra cpu` or `--extra cu126`); without it, it re-syncs the environment and removes torch.

Each run writes `results/<system>/max_norm_<m>/seed_<s>/<condition>.json` (configuration, validation MSE per model and per step, training logs) and `<condition>-models.pt` (state dicts and normalisation multipliers, not tracked by git).
`--merge` reads every run and writes:

- `results/summary.json`: validation MSE per seed, geometric mean, min and max, per system, max norm, condition and model.
- `results/validation.png`: validation MSE per seed and geometric mean, against max norm.
- `results/<system>/clipping-max_norm_<m>.png`: training MSE, gradient norm and clipping scale per epoch; rows are conditions with shared y-axes, the line is the mean over seeds (geometric on log axes) and the band covers min to max, after a 25-epoch rolling median.

`--data` and `--out-dir` override the default paths.

### SLURM

`clip_sweep.sbatch` runs one array task per line of `sweep_runs.txt`, one GPU each.
`make_sweep_runs.py` writes that list: 2 systems x seeds 0–2 x max norm {1, 10, 100, inf} x {joint, separate}, without the 4 runs already in `results/` (seed 0, max norm 1), so 44 lines.
From the repository root, after `uv sync --extra cu126` and generating both datasets:

```bash
sbatch analysis/gradient_clipping/clip_sweep.sbatch              # all 44 runs
sbatch -a 1-44%20 analysis/gradient_clipping/clip_sweep.sbatch   # at most 20 at a time
# afterwards, on any node:
uv run --extra cu126 python analysis/gradient_clipping/clip_test_1d.py --merge
```

Arguments after the script name go to `clip_test_1d.py`, e.g. a one-epoch check: `sbatch -a 1 analysis/gradient_clipping/clip_sweep.sbatch --epochs 1`.
Account, partition, runtime and resources are in the `#SBATCH` lines.

## Results (seed 0, max norm 1)

Joint clipping is not dominated by the model with the lowest loss but by FRNN, whose gradient norm (median about 31–37) is about 90% of the joint norm squared on both systems, under both conditions.
Separate clipping improves REF (validation MSE 0.646 to 0.493 on the string, 0.982 to 0.570 on the nonlinear string), but REF stays about 10x worse than FGRU and FRNN.
On the nonlinear string, separate clipping improves FGRU 3.4x (2.95e-2 to 8.67e-3).
The sweep over seeds and max norms tests whether these differences hold.

## Expected signature

If joint clipping explains the poor REF results, under `joint` REF's clipping scale stays well below 1 while FGRU's gradient norm dominates the joint norm, and REF's training and validation MSE improve under `separate`.
