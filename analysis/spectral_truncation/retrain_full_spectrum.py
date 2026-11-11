"""Reduced-scale retraining of the 2D FGRU/FRNN with and without the full-spectrum fix.

Follows the 2D wave set-up of train_2d_wave.py at commit e08b4b0 (40 x 38 grid, 48 kHz,
dur = 0.0007 s, width 8, AdamW + OneCycleLR, log10 MSE loss), with fewer examples and
epochs so that it runs on a CPU. Each model has its own optimiser.

Usage:
  uv run python analysis/spectral_truncation/retrain_full_spectrum.py [--epochs N] ...
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from dafx22_fno.generators.wave_solver_2d import WaveSolver2D  # noqa: E402
from dafx22_fno.modules.fno_gru import FNO_GRU_2d  # noqa: E402
from dafx22_fno.modules.fno_ref import FNO_Markov_2d  # noqa: E402
from dafx22_fno.modules.fno_rnn import FNO_RNN_2d  # noqa: E402
from full_spectrum import use_full_spectrum  # noqa: E402

FS = 48000
NUM_POINTS = 40
SIMULATED_MODES = 25
ROOM_SIZE = 1.0
ASPECT = 0.95
DUR = 0.0007
WIDTH = 8

MODEL_STYLE = {
    "FGRU": ("#eb6834", "-"),
    "FGRU full spectrum": ("#eb6834", "--"),
    "FRNN": ("#1baf7a", "-"),
    "FRNN full spectrum": ("#1baf7a", "--"),
    "REF": ("#2a78d6", "-"),
}


def make_solver(dur):
    return WaveSolver2D(
        dur=dur,
        Fs=FS,
        lx=ROOM_SIZE,
        ly=ASPECT * ROOM_SIZE,
        spatial_delta=ROOM_SIZE / NUM_POINTS,
        modes=SIMULATED_MODES,
    )


def generate(num, dur, rng_seed):
    """Half impulses at random positions, half random initial conditions (as e08b4b0)."""
    np.random.seed(rng_seed)
    solver = make_solver(dur)
    inputs = torch.zeros((num, solver.numXs, solver.numYs, 3))
    outputs = torch.zeros((num, solver.numT - 1, solver.numXs, solver.numYs, 3))
    for i in range(num):
        if i < num // 2:
            fe_x = solver.create_impulse(np.random.rand(1), np.random.rand(1))
        else:
            fe_x = solver.create_random_initial()
        _, _, y_sp, y_vx, y_vy = solver.solve(fe_x)
        fields = np.stack([y_sp, y_vx, y_vy], axis=-1)  # (x, y, t, c)
        inputs[i] = torch.tensor(fields[:, :, 0])
        outputs[i] = torch.tensor(fields[:, :, 1:].transpose(2, 0, 1, 3))
    return inputs, outputs


def load_or_generate(args):
    """Normalised (x_train, y_train, x_val, y_val); cached so parallel runs share data."""
    cache = args.data_cache
    if cache is not None and cache.exists():
        return torch.load(cache)
    x_train, y_train = generate(args.variations, DUR, args.seed)
    x_val, y_val = generate(args.validation, 2 * DUR, args.seed + 1)
    norm = 1 / torch.std(y_train, dim=(0, 1, 2, 3))
    data = tuple(t * norm for t in (x_train, y_train, x_val, y_val))
    if cache is not None:
        torch.save(data, cache)
    return data


def build_models(nx, ny, seed):
    def seeded(factory):
        torch.manual_seed(seed)
        return factory()

    gru = seeded(lambda: FNO_GRU_2d(3, 3, nx, ny, width=WIDTH))
    rnn = seeded(lambda: FNO_RNN_2d(3, 3, nx, ny, width=WIDTH, depth=3))
    models = {
        "FGRU": gru,
        # Same initial positive-kx weights; the negative-kx block is new.
        "FGRU full spectrum": use_full_spectrum(copy.deepcopy(gru)),
        "FRNN": rnn,
        "FRNN full spectrum": use_full_spectrum(copy.deepcopy(rnn)),
        "REF": seeded(lambda: FNO_Markov_2d(3, 3, nx, ny, WIDTH, depth=3)),
    }
    return models


def train(model, data, epochs, batch_size, seed):
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader)
    )
    history = np.zeros(epochs)
    for ep in range(epochs):
        tic = time.time()
        total, count = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x, num_time_steps=y.shape[1])
            mse = torch.nn.functional.mse_loss(pred, y)
            torch.log10(mse).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            total += float(mse.detach()) * len(x)
            count += len(x)
        history[ep] = total / count
        print(
            f"\r  epoch {ep + 1}/{epochs} train MSE {history[ep]:.3e} "
            f"({time.time() - tic:.1f} s/epoch)",
            end="",
            flush=True,
        )
    print()
    return history


@torch.no_grad()
def evaluate(model, x, y):
    pred = model(x, num_time_steps=y.shape[1])
    per_step = ((pred - y) ** 2).mean(dim=(0, 2, 3, 4))
    return float(per_step.mean()), per_step.numpy()


def plot_results(histories, rollouts, train_steps, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for name, hist in histories.items():
        color, ls = MODEL_STYLE[name]
        axes[0].plot(np.arange(1, len(hist) + 1), hist, ls, color=color, lw=2, label=name)
        axes[1].plot(
            np.arange(1, len(rollouts[name]) + 1),
            rollouts[name],
            ls,
            color=color,
            lw=2,
            label=name,
        )
    axes[0].set(xlabel="epoch", ylabel="training MSE", yscale="log", title="Training")
    axes[1].axvline(train_steps, color="#52514e", lw=0.8, ls=":")
    axes[1].set(
        xlabel="time step",
        ylabel="validation MSE",
        yscale="log",
        title="Validation rollout (66 steps)",
    )
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def merge(tags):
    merged = {"config": {}, "results": {}, "histories": {}, "rollouts": {}}
    for tag in tags:
        part = json.loads((HERE / f"retrain_results_{tag}.json").read_text())
        merged["config"] = part["config"] | {"models": list(MODEL_STYLE)}
        for key in ("results", "histories", "rollouts"):
            merged[key] |= part[key]
    order = [m for m in MODEL_STYLE if m in merged["results"]]
    for key in ("results", "histories", "rollouts"):
        merged[key] = {m: merged[key][m] for m in order}
    (HERE / "retrain_results.json").write_text(json.dumps(merged, indent=2) + "\n")
    plot_results(
        {k: np.asarray(v) for k, v in merged["histories"].items()},
        {k: np.asarray(v) for k, v in merged["rollouts"].items()},
        merged["config"]["train_steps"],
        HERE / "figures" / "fig7_retrain.png",
    )
    for tag in tags:
        (HERE / f"retrain_results_{tag}.json").unlink()
        (HERE / "figures" / f"fig7_retrain_{tag}.png").unlink(missing_ok=True)
    print(json.dumps(merged["results"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variations", type=int, default=256)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=list(MODEL_STYLE))
    parser.add_argument("--tag", default="")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="truncate training rollouts to this many steps (default: all 33)",
    )
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="TAG",
        help="merge retrain_results_<TAG>.json files into retrain_results.json and plot",
    )
    args = parser.parse_args()
    if args.merge:
        merge(args.merge)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.generate_only:
        load_or_generate(args)
        return

    tic = time.time()
    x_train, y_train, x_val, y_val = load_or_generate(args)
    y_train = y_train[:, : args.train_steps]
    train_steps = y_train.shape[1]
    print(
        f"data: train {tuple(y_train.shape)}, validation {tuple(y_val.shape)} "
        f"({time.time() - tic:.1f} s)"
    )

    models = build_models(x_train.shape[1], x_train.shape[2], args.seed)
    results, histories, rollouts = {}, {}, {}
    for name in args.models:
        model = models[name]
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {n_params} parameters")
        tic = time.time()
        histories[name] = train(
            model, (x_train, y_train), args.epochs, args.batch_size, args.seed
        )
        _, per_step = evaluate(model, x_val, y_val)
        rollouts[name] = per_step
        results[name] = {
            "parameters": n_params,
            "final_train_mse": float(histories[name][-1]),
            "val_mse_training_length": float(per_step[:train_steps].mean()),
            "val_mse_33_steps": float(per_step[:33].mean()),
            "val_mse_2x_length": float(per_step.mean()),
            "train_time_s": time.time() - tic,
        }
        print(f"  {results[name]}")

    suffix = f"_{args.tag}" if args.tag else ""
    out = {"config": vars(args) | {"train_steps": train_steps}, "results": results}
    out["histories"] = {k: v.tolist() for k, v in histories.items()}
    out["rollouts"] = {k: v.tolist() for k, v in rollouts.items()}
    (HERE / f"retrain_results{suffix}.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "figures").mkdir(exist_ok=True)
    plot_results(
        histories, rollouts, train_steps, HERE / "figures" / f"fig7_retrain{suffix}.png"
    )


if __name__ == "__main__":
    main()
