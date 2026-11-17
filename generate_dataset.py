"""Simulate a dataset of the DAFX22 paper and save it as one .npy array.

Writes the raw (unnormalised) array (trajectories, time steps, x, [y,] channels) and a
.json next to it with the solver settings, seed, channel names and the per-channel
mean and std of the training split (all but the last ceil(10%) trajectories).

Usage:
  uv run python generate_dataset.py {string,nonlinear_string,wave2d} [--variations N] [--seed S] [--steps T] [--output PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dafx22_fno.datasets import SYSTEMS, settings, simulate, split


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("--variations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="time steps per trajectory, input included (default: the paper's duration)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: output/datasets/<system>-n_<variations>[-steps_<steps>]-seed_<seed>.npy",
    )
    args = parser.parse_args()
    steps = f"-steps_{args.steps}" if args.steps is not None else ""
    output = args.output or Path(
        f"output/datasets/{args.system}-n_{args.variations}{steps}-seed_{args.seed}.npy"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    data = simulate(args.system, args.variations, args.seed, args.steps)
    np.save(output, data)

    training, _ = split(data, args.val_split)
    axes = tuple(range(training.ndim - 1))
    info = {
        "system": args.system,
        "shape": list(data.shape),
        "channels": SYSTEMS[args.system]["channels"],
        "seed": args.seed,
        "val_split": args.val_split,
        "settings": settings(args.system, args.steps),
        "training_split": {
            "mean": training.mean(axis=axes).tolist(),
            "std": training.std(axis=axes).tolist(),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Dataset saved: {output} {data.shape}")


if __name__ == "__main__":
    main()
