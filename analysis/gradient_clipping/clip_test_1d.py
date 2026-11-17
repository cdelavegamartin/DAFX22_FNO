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

Usage:
  uv run python analysis/gradient_clipping/clip_test_1d.py --condition joint --tag joint
  uv run python analysis/gradient_clipping/clip_test_1d.py --condition separate --tag separate
  uv run python analysis/gradient_clipping/clip_test_1d.py --merge joint separate
"""

import argparse
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
sys.path.insert(0, str(HERE.parents[1]))

from dafx22_fno.generators.string_solver import StringSolver  # noqa: E402
from dafx22_fno.modules.fno_gru import FNO_GRU_1d  # noqa: E402
from dafx22_fno.modules.fno_ref import FNO_Markov_1d  # noqa: E402
from dafx22_fno.modules.fno_rnn import FNO_RNN_1d  # noqa: E402

# Settings of train_1d_string.py at commit e08b4b0
DUR, FS, DELTA_X, D1 = 0.0025, 48000, 1e-2, 1e-1
WIDTH = 16
LR, MAX_LR, WEIGHT_DECAY, MAX_NORM = 2e-4, 1e-3, 1e-4, 1.0
NAMES = ("gru", "rnn", "ref")
LABELS = {"gru": "FGRU", "rnn": "FRNN", "ref": "REF"}
COLORS = {"gru": "#eb6834", "rnn": "#1baf7a", "ref": "#2a78d6"}
STYLES = {"joint": "-", "separate": "--"}


def generate(num, seed):
    """Half plucks at random positions, half random initial conditions (as e08b4b0)."""
    np.random.seed(seed)
    solver = StringSolver(dur=DUR, Fs=FS, delta_x=DELTA_X, d1=D1)
    inputs = torch.zeros((num, 1, solver.numXs, 2))
    outputs = torch.zeros((num, solver.numT - 1, solver.numXs, 2))
    for i in range(num):
        fe_x = (
            solver.create_pluck(np.random.rand(1))
            if i < num // 2
            else solver.create_random_initial()
        )
        y_x, y_defl_x = solver.solve(fe_x)
        inputs[i, 0] = torch.tensor(np.stack([y_x[:, 0], y_defl_x[:, 0]], axis=-1))
        outputs[i] = torch.tensor(
            np.stack([y_x[:, 1:].T, y_defl_x[:, 1:].T], axis=-1)
        )
    norms = 1 / torch.std(outputs, dim=(0, 1, 2))
    inputs, outputs = inputs * norms, outputs * norms
    num_val = int(np.ceil(0.1 * num))
    return inputs[:-num_val], outputs[:-num_val], inputs[-num_val:], outputs[-num_val:]


def load_or_generate(path, num, seed):
    if path is not None and path.exists():
        return torch.load(path)
    data = generate(num, seed)
    if path is not None:
        torch.save(data, path)
    return data


def build_models(size, seed):
    """FGRU, FRNN, REF in the order and seeding of e08b4b0."""
    torch.manual_seed(seed)
    return {
        "gru": FNO_GRU_1d(2, 2, size, width=WIDTH),
        "rnn": FNO_RNN_1d(2, 2, size, depth=3, width=WIDTH),
        "ref": FNO_Markov_1d(2, 2, size, depth=3, width=WIDTH),
    }


def grad_norm(params):
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads)))


def make_optimisers(models, condition, epochs, steps_per_epoch):
    """One optimiser over all parameters (joint) or one per model (separate)."""
    groups = (
        [[p for m in models.values() for p in m.parameters()]]
        if condition == "joint"
        else [list(m.parameters()) for m in models.values()]
    )
    optimisers = [torch.optim.AdamW(g, lr=LR, weight_decay=WEIGHT_DECAY) for g in groups]
    schedulers = [
        torch.optim.lr_scheduler.OneCycleLR(
            o, max_lr=MAX_LR, epochs=epochs, steps_per_epoch=steps_per_epoch
        )
        for o in optimisers
    ]
    return groups, optimisers, schedulers


def train(models, data, condition, epochs, batch_size, seed, device):
    x_train, y_train = data
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    groups, optimisers, schedulers = make_optimisers(
        models, condition, epochs, len(loader)
    )
    log = {key: {n: np.zeros(epochs) for n in NAMES} for key in ("mse", "norm", "scale")}
    log["joint_norm"] = np.zeros(epochs)
    for ep in range(epochs):
        tic = time.time()
        sums = {key: {n: 0.0 for n in NAMES} for key in ("mse", "norm", "scale")}
        joint_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for o in optimisers:
                o.zero_grad()
            for name, model in models.items():
                loss = torch.log10(
                    torch.nn.functional.mse_loss(model(x[:, 0], num_time_steps=y.shape[1]), y)
                )
                loss.backward()
                sums["mse"][name] += 10 ** loss.item()
            # Gradient norms before clipping
            norms = {n: grad_norm(m.parameters()) for n, m in models.items()}
            joint = float(np.sqrt(sum(v**2 for v in norms.values())))
            for group in groups:
                torch.nn.utils.clip_grad_norm_(group, MAX_NORM)
            for name in models:
                applied = joint if condition == "joint" else norms[name]
                sums["norm"][name] += norms[name]
                sums["scale"][name] += min(1.0, MAX_NORM / applied)
            joint_sum += joint
            for o in optimisers:
                o.step()
            for s in schedulers:
                s.step()
        for key in sums:
            for name in models:
                log[key][name][ep] = sums[key][name] / len(loader)
        log["joint_norm"][ep] = joint_sum / len(loader)
        mse = ", ".join(f"{n} {log['mse'][n][ep]:.2e}" for n in models)
        print(
            f"\r  {condition} epoch {ep + 1}/{epochs}: {mse} "
            f"({time.time() - tic:.1f} s/epoch)",
            end="",
            flush=True,
        )
    print()
    return log


@torch.no_grad()
def validate(models, x_val, y_val, device):
    x_val, y_val = x_val.to(device), y_val.to(device)
    out = {}
    for name, model in models.items():
        err = (model(x_val[:, 0], num_time_steps=y_val.shape[1]) - y_val) ** 2
        out[name] = {
            "val_mse": float(err.mean()),
            "val_mse_per_step": err.mean(dim=(0, 2, 3)).tolist(),
        }
    return out


def to_lists(log):
    return {
        k: ({n: v.tolist() for n, v in val.items()} if isinstance(val, dict) else val.tolist())
        for k, val in log.items()
    }


def plot(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), layout="constrained")
    panels = (
        ("mse", "training MSE", True),
        ("norm", "gradient norm before clipping", True),
        ("scale", "gradient scale applied by clipping", False),
    )
    for ax, (key, ylabel, log_y) in zip(axes, panels):
        for condition, res in results.items():
            for name in NAMES:
                ax.plot(
                    np.arange(1, len(res["log"][key][name]) + 1),
                    res["log"][key][name],
                    STYLES[condition],
                    color=COLORS[name],
                    lw=1.5,
                    label=f"{LABELS[name]}, {condition}",
                )
        ax.set(xlabel="epoch", ylabel=ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside right center",
        frameon=False,
        fontsize=7,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def merge(tags):
    results = {}
    for tag in tags:
        part = json.loads((HERE / f"results_{tag}.json").read_text())
        results[part["config"]["condition"]] = part
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (HERE / "figures").mkdir(exist_ok=True)
    plot(results, HERE / "figures" / "clipping_1d.png")
    for tag in tags:
        (HERE / f"results_{tag}.json").unlink()
    for condition, res in results.items():
        summary = ", ".join(
            f"{LABELS[n]} {res['validation'][n]['val_mse']:.3e}" for n in NAMES
        )
        print(f"{condition}: validation MSE {summary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--condition", choices=("joint", "separate"))
    parser.add_argument("--variations", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--merge", nargs="+", metavar="TAG")
    args = parser.parse_args()
    if args.merge:
        merge(args.merge)
        return
    if args.threads:
        torch.set_num_threads(args.threads)

    x_train, y_train, x_val, y_val = load_or_generate(
        args.data_cache, args.variations, args.seed
    )
    if args.generate_only:
        return
    print(f"data: train {tuple(y_train.shape)}, validation {tuple(y_val.shape)}")
    models = {n: m.to(args.device) for n, m in build_models(x_train.shape[2], args.seed).items()}
    log = train(
        models,
        (x_train, y_train),
        args.condition,
        args.epochs,
        args.batch_size,
        args.seed,
        args.device,
    )
    result = {
        "config": {
            k: getattr(args, k)
            for k in ("condition", "variations", "epochs", "batch_size", "seed")
        },
        "validation": validate(models, x_val, y_val, args.device),
        "log": to_lists(log),
    }
    tag = args.tag or args.condition
    (HERE / f"results_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
