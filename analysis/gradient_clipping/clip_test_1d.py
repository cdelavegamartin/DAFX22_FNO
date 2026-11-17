"""Joint vs separate gradient clipping for the 1D string models of Parker et al. (2022).

train_1d_string.py (commit e08b4b0) trains FGRU, FRNN and REF with one AdamW over all
their parameters and clips the norm of their concatenated gradients to 1.
The loss is log10(MSE), whose gradient is grad(MSE) / (MSE ln 10), so the model with
the lowest loss dominates the joint norm and the clip shrinks the steps of the others.

This script trains the three models under two conditions that differ only in clipping:
  joint     one AdamW and one OneCycleLR over all parameters, joint clipping (paper)
  separate  one AdamW and one OneCycleLR per model, each model clipped on its own
Both start from the same initial weights and see the same batches.
AdamW and OneCycleLR act per parameter, so the optimisers are equivalent across conditions.

Usage (system: string or nonlinear_string):
  uv run python generate_dataset.py string
  uv run python analysis/gradient_clipping/clip_test_1d.py --system string --condition joint
  uv run python analysis/gradient_clipping/clip_test_1d.py --system string --condition separate
  uv run python analysis/gradient_clipping/clip_test_1d.py --merge
Each run writes <condition>.json and <condition>-models.pt (state dicts and
normalisation multipliers) to results/<system>/max_norm_<m>/seed_<s>; --merge reads
every run and writes results/summary.json (validation MSE per seed, geometric mean,
min, max), results/validation.png and results/<system>/clipping-max_norm_<m>.png.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from dafx22_fno.datasets import split  # noqa: E402
from dafx22_fno.modules.fno_gru import FNO_GRU_1d  # noqa: E402
from dafx22_fno.modules.fno_ref import FNO_Markov_1d  # noqa: E402
from dafx22_fno.modules.fno_rnn import FNO_RNN_1d  # noqa: E402
from dafx22_fno.sweep_plots import MODEL_LABELS, plot_training, plot_validation  # noqa: E402
from dafx22_fno.training import (  # noqa: E402
    CONDITIONS,
    Hyperparameters,
    save_models,
    to_lists,
    train,
    validate,
)

# Model width of train_1d_string.py and train_1d_nonlinear_string.py at commit
# e08b4b0; the other settings are the same for both
WIDTHS = {"string": 16, "nonlinear_string": 32}
HYPERPARAMETERS = Hyperparameters(lr=2e-4, max_lr=1e-3, weight_decay=1e-4, max_norm=1.0)
NAMES = ("gru", "rnn", "ref")
LABELS = MODEL_LABELS


def load(path, val_split=0.1):
    """Normalised (x_train, y_train, x_val, y_val) and norms from a generate_dataset.py array.

    As in e08b4b0, each channel is scaled by the std of the output steps of all
    trajectories, and the last ceil(val_split * n) trajectories validate.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; create it with: "
            "uv run python generate_dataset.py string --output " + str(path)
        )
    data = torch.from_numpy(np.load(path))
    inputs, outputs = data[:, :1], data[:, 1:]
    norms = 1 / torch.std(outputs, dim=(0, 1, 2))
    inputs, outputs = inputs * norms, outputs * norms
    (x_train, x_val), (y_train, y_val) = split(inputs, val_split), split(outputs, val_split)
    return x_train, y_train, x_val, y_val, norms


def build_models(size, seed, width):
    """FGRU, FRNN, REF in the order and seeding of e08b4b0."""
    torch.manual_seed(seed)
    return {
        "gru": FNO_GRU_1d(2, 2, size, width=width),
        "rnn": FNO_RNN_1d(2, 2, size, depth=3, width=width),
        "ref": FNO_Markov_1d(2, 2, size, depth=3, width=width),
    }


def load_runs(results_dir):
    """{system: {max norm: {condition: [run per seed]}}} from results/<system>/max_norm_*/seed_*/."""
    runs = {}
    for path in sorted(results_dir.glob("*/max_norm_*/seed_*/*.json")):
        run = json.loads(path.read_text())
        config = run["config"]
        max_norm = float(config.get("max_norm", 1.0))  # the first runs predate --max-norm
        by_condition = runs.setdefault(config["system"], {}).setdefault(max_norm, {})
        by_condition.setdefault(config["condition"], []).append(run)
    return {
        system: dict(sorted(by_norm.items())) for system, by_norm in sorted(runs.items())
    }


def merge(results_dir):
    """Figures and summary.json over seeds of every run in results_dir."""
    runs = load_runs(results_dir)
    summary, validation = {}, {}
    for system, by_norm in runs.items():
        validation[system] = {}
        for max_norm, by_condition in by_norm.items():
            rows = [
                (condition, [run["log"] for run in by_condition[condition]], condition == "joint")
                for condition in CONDITIONS
                if condition in by_condition
            ]
            plot_training(
                rows,
                results_dir / system / f"clipping-max_norm_{max_norm:g}.png",
                f"{system}, max norm {max_norm:g}",
            )
            validation[system][f"{max_norm:g}"] = {
                condition: {
                    name: [run["validation"][name]["val_mse"] for run in by_condition[condition]]
                    for name in NAMES
                }
                for condition in CONDITIONS
                if condition in by_condition
            }
            summary.setdefault(system, {})[f"{max_norm:g}"] = {
                condition: {
                    "seeds": [run["config"]["seed"] for run in by_condition[condition]],
                    **{
                        name: {
                            "val_mse": values,
                            "geometric_mean": float(np.exp(np.mean(np.log(values)))),
                            "min": min(values),
                            "max": max(values),
                        }
                        for name, values in by_model.items()
                    },
                }
                for condition, by_model in validation[system][f"{max_norm:g}"].items()
            }
    plot_validation(validation, results_dir / "validation.png", "max norm")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for system, by_norm in summary.items():
        for max_norm, by_condition in by_norm.items():
            for condition, res in by_condition.items():
                line = ", ".join(
                    f"{LABELS[n]} {res[n]['geometric_mean']:.3e} "
                    f"[{res[n]['min']:.1e}, {res[n]['max']:.1e}]"
                    for n in NAMES
                )
                print(f"{system}, max norm {max_norm}, {condition} "
                      f"(seeds {res['seeds']}): {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", choices=sorted(WIDTHS), default="string")
    parser.add_argument("--condition", choices=("joint", "separate"))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-norm",
        type=float,
        default=HYPERPARAMETERS.max_norm,
        help="gradient clipping threshold; inf disables clipping",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset from generate_dataset.py "
        "(default: output/datasets/<system>-n_1024-seed_0.npy)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: analysis/gradient_clipping/results/<system>/"
        "max_norm_<max norm>/seed_<seed>",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="figures and summary over all runs in --out-dir (default: results/)",
    )
    args = parser.parse_args()
    if args.merge:
        merge(args.out_dir or HERE / "results")
        return
    data_path = args.data or Path(f"output/datasets/{args.system}-n_1024-seed_0.npy")
    out_dir = args.out_dir or (
        HERE / "results" / args.system / f"max_norm_{args.max_norm:g}" / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.condition is None:
        parser.error("--condition is required unless --merge is given")
    if args.threads:
        torch.set_num_threads(args.threads)

    x_train, y_train, x_val, y_val, norms = load(data_path)
    print(f"data: train {tuple(y_train.shape)}, validation {tuple(y_val.shape)}")
    models = build_models(x_train.shape[2], args.seed, WIDTHS[args.system])
    models = {n: m.to(args.device) for n, m in models.items()}
    log = train(
        models,
        (x_train, y_train),
        args.condition,
        dataclasses.replace(HYPERPARAMETERS, max_norm=args.max_norm),
        args.epochs,
        args.batch_size,
        args.seed,
        args.device,
    )
    result = {
        "config": {
            "system": args.system,
            "condition": args.condition,
            "data": str(data_path),
            "width": WIDTHS[args.system],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_norm": args.max_norm,
        },
        "validation": validate(models, x_val, y_val, args.device),
        "log": to_lists(log),
    }
    (out_dir / f"{args.condition}.json").write_text(json.dumps(result, indent=2) + "\n")
    save_models(models, norms, out_dir / f"{args.condition}-models.pt")


if __name__ == "__main__":
    main()
