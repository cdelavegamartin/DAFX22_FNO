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

Each condition needs about 8 GB of memory (autograd over 119 steps at batch 512); on 14 CPU threads an epoch takes about 30 s, so the paper's 5000 epochs need a GPU.

```bash
uv sync --extra cu126
uv run python analysis/gradient_clipping/clip_test_1d.py --generate-only --data-cache string1d.pt
uv run python analysis/gradient_clipping/clip_test_1d.py --condition joint --data-cache string1d.pt --device cuda
uv run python analysis/gradient_clipping/clip_test_1d.py --condition separate --data-cache string1d.pt --device cuda
uv run python analysis/gradient_clipping/clip_test_1d.py --merge joint separate
```

The two conditions are independent and MAY run as parallel jobs.
`--merge` writes `results.json` (validation MSE per model, per-step validation MSE, training logs) and `figures/clipping_1d.png` (training MSE, gradient norms and clipping scale per model and condition).

## Expected signature

If joint clipping explains the poor REF results, under `joint` REF's clipping scale stays well below 1 while FGRU's gradient norm dominates the joint norm, and REF's training and validation MSE improve under `separate`.
