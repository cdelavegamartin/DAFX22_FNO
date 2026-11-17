"""Full-scale 2D wave training with the original or the full-spectrum FourierConv2d (issue #2).

Follows train_2d_wave.py at commit e08b4b0 (Parker et al., 2022): FGRU, FRNN and REF at
width 8, one AdamW (lr 1e-4, weight decay 1e-4) and one OneCycleLR (max 1e-3) over all
their parameters, joint gradient clipping at 1, log10(MSE) loss, batch 400, 5000 epochs,
33 output steps.
`--layer full` replaces the FourierConv2d layers of FGRU and FRNN by
FourierConv2dFullSpectrum, keeping the original initial weights for kx >= 0; REF has no
FourierConv2d and starts from the same weights in both layers.
The models train on the first 33 output steps of a longer dataset and validate on its
last 10% of trajectories, over 33 steps (the paper's metric) and per step over all steps.

Usage:
  uv run python generate_dataset.py wave2d --steps 67   # 66 output steps
  uv run python analysis/spectral_truncation/wave2d_sweep.py --layer original --seed 0 --device cuda
  uv run python analysis/spectral_truncation/wave2d_sweep.py --merge
Each run writes run.json and models.pt (state dicts and normalisation multipliers) to
wave2d_sweep/layer_<layer>/seed_<seed>; --merge reads every run and writes
wave2d_sweep/summary.json, validation.png, validation_per_step.png and training.png.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from dafx22_fno.datasets import load_normalised  # noqa: E402
from dafx22_fno.modules.fno_gru import FNO_GRU_2d  # noqa: E402
from dafx22_fno.modules.fno_ref import FNO_Markov_2d  # noqa: E402
from dafx22_fno.modules.fno_rnn import FNO_RNN_2d  # noqa: E402
from dafx22_fno.sweep_plots import (  # noqa: E402
    MODEL_LABELS,
    plot_per_step,
    plot_training,
    plot_validation,
)
from dafx22_fno.training import (  # noqa: E402
    Hyperparameters,
    save_models,
    to_lists,
    train,
    validate,
)
from full_spectrum import use_full_spectrum  # noqa: E402

WIDTH = 8
TRAIN_STEPS = 33
HYPERPARAMETERS = Hyperparameters(lr=1e-4, max_lr=1e-3, weight_decay=1e-4, max_norm=1.0)
LAYERS = ("original", "full")
LAYER_LABELS = {"original": "original layer", "full": "full-spectrum layer"}
NAMES = ("gru", "rnn", "ref")
# Validation MSE of Parker et al. (2022), table of the 2D wave equation
PAPER_VAL_MSE = {"gru": 1.86e-2, "rnn": 1.54e-2, "ref": 1.12e-4}
RESULTS = HERE / "wave2d_sweep"


def with_full_spectrum(model):
    """Copy of `model` with full-spectrum layers and the same weights for kx >= 0."""
    full = use_full_spectrum(copy.deepcopy(model))
    missing, unexpected = full.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all(k.endswith(("_neg")) for k in missing), (missing, unexpected)
    return full


def build_models(nx, ny, seed, layer):
    """FGRU, FRNN, REF in the order of e08b4b0, seeded; FGRU and FRNN converted for `full`."""
    torch.manual_seed(seed)
    models = {
        "gru": FNO_GRU_2d(3, 3, nx, ny, width=WIDTH),
        "rnn": FNO_RNN_2d(3, 3, nx, ny, depth=3, width=WIDTH),
        "ref": FNO_Markov_2d(3, 3, nx, ny, depth=3, width=WIDTH),
    }
    if layer == "full":
        models["gru"] = with_full_spectrum(models["gru"])
        models["rnn"] = with_full_spectrum(models["rnn"])
    return models


def run(args):
    out_dir = args.out_dir or RESULTS / f"layer_{args.layer}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val, norms = load_normalised(args.data, train_steps=TRAIN_STEPS)
    print(f"data: train {tuple(y_train.shape)}, validation {tuple(y_val.shape)}")
    models = build_models(x_train.shape[2], x_train.shape[3], args.seed, args.layer)
    models = {n: m.to(args.device) for n, m in models.items()}
    log = train(
        models,
        (x_train, y_train),
        "joint",
        HYPERPARAMETERS,
        args.epochs,
        args.batch_size,
        args.seed,
        args.device,
    )
    result = {
        "config": {
            "layer": args.layer,
            "data": str(args.data),
            "width": WIDTH,
            "train_steps": TRAIN_STEPS,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "parameters": {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()},
        },
        "validation": validate(models, x_val, y_val, args.device, train_steps=TRAIN_STEPS),
        "log": to_lists(log),
    }
    (out_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    save_models(models, norms, out_dir / "models.pt")


def merge(results_dir):
    """Figures and summary.json over seeds of every run in results_dir."""
    runs = {layer: [] for layer in LAYERS}
    for path in sorted(results_dir.glob("layer_*/seed_*/run.json")):
        result = json.loads(path.read_text())
        runs[result["config"]["layer"]].append(result)
    runs = {layer: rs for layer, rs in runs.items() if rs}
    val_mse = {
        layer: {n: [r["validation"][n]["val_mse"] for r in rs] for n in NAMES}
        for layer, rs in runs.items()
    }
    summary = {
        "paper": PAPER_VAL_MSE,
        **{
            layer: {
                "seeds": [r["config"]["seed"] for r in runs[layer]],
                **{
                    n: {
                        "val_mse": values,
                        "geometric_mean": float(np.exp(np.mean(np.log(values)))),
                        "min": min(values),
                        "max": max(values),
                    }
                    for n, values in by_model.items()
                },
            }
            for layer, by_model in val_mse.items()
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_validation(
        {f"2D wave, validation MSE over {TRAIN_STEPS} steps": {
            LAYER_LABELS[layer]: {"joint clipping": by_model} for layer, by_model in val_mse.items()
        }},
        results_dir / "validation.png",
        "FourierConv2d",
    )
    plot_per_step(
        {
            LAYER_LABELS[layer]: {
                n: [r["validation"][n]["val_mse_per_step"] for r in rs] for n in NAMES
            }
            for layer, rs in runs.items()
        },
        results_dir / "validation_per_step.png",
        TRAIN_STEPS,
    )
    plot_training(
        [(LAYER_LABELS[layer], [r["log"] for r in rs], True) for layer, rs in runs.items()],
        results_dir / "training.png",
        "2D wave, joint clipping at 1",
    )
    print("| model | paper | " + " | ".join(LAYER_LABELS[layer] for layer in runs) + " |")
    print("|---|---|" + "---|" * len(runs))
    for n in NAMES:
        cells = [
            f"{summary[layer][n]['geometric_mean']:.2e} "
            f"[{summary[layer][n]['min']:.1e}, {summary[layer][n]['max']:.1e}]"
            for layer in runs
        ]
        print(f"| {MODEL_LABELS[n]} | {PAPER_VAL_MSE[n]:.2e} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layer", choices=LAYERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("output/datasets/wave2d-n_1024-steps_67-seed_0.npy"),
        help="dataset from generate_dataset.py with at least 34 time steps",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="run: default wave2d_sweep/layer_<layer>/seed_<seed>; "
        "--merge: default wave2d_sweep/",
    )
    parser.add_argument("--merge", action="store_true", help="figures and summary over all runs")
    args = parser.parse_args()
    if args.merge:
        merge(args.out_dir or RESULTS)
    elif args.layer is None:
        parser.error("--layer is required unless --merge is given")
    else:
        run(args)


if __name__ == "__main__":
    main()
