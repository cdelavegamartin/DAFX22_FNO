"""Training loop of the DAFX22 scripts (Parker et al., 2022) with joint or separate clipping.

The paper's scripts (commit e08b4b0) train FGRU, FRNN and REF together: one AdamW and
one OneCycleLR over all their parameters, a log10(MSE) loss per model, and one clip of
the norm of the concatenated gradients (`joint`).
`separate` gives each model its own optimiser, scheduler and clip; AdamW and OneCycleLR
act per parameter, so clipping is the only difference between the two.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

CONDITIONS = ("joint", "separate")


@dataclass(frozen=True)
class Hyperparameters:
    lr: float
    max_lr: float = 1e-3
    weight_decay: float = 1e-4
    max_norm: float = 1.0  # inf disables clipping


def grad_norm(params):
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads)))


def make_optimisers(models, condition, hp, epochs, steps_per_epoch):
    """Parameter groups, optimisers and schedulers: one for all models (joint) or one per model."""
    groups = (
        [[p for m in models.values() for p in m.parameters()]]
        if condition == "joint"
        else [list(m.parameters()) for m in models.values()]
    )
    optimisers = [
        torch.optim.AdamW(g, lr=hp.lr, weight_decay=hp.weight_decay) for g in groups
    ]
    schedulers = [
        torch.optim.lr_scheduler.OneCycleLR(
            o, max_lr=hp.max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch
        )
        for o in optimisers
    ]
    return groups, optimisers, schedulers


def train(models, data, condition, hp, epochs, batch_size, seed, device):
    """Train `models` (dict name -> module) in place; returns per-epoch logs.

    Logs per model the mean over batches of the training MSE, the gradient norm before
    clipping and the factor by which clipping scales the gradient, plus the joint norm.
    """
    x_train, y_train = data
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    groups, optimisers, schedulers = make_optimisers(
        models, condition, hp, epochs, len(loader)
    )
    names = list(models)
    log = {key: {n: np.zeros(epochs) for n in names} for key in ("mse", "norm", "scale")}
    log["joint_norm"] = np.zeros(epochs)
    for ep in range(epochs):
        tic = time.time()
        sums = {key: {n: 0.0 for n in names} for key in ("mse", "norm", "scale")}
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
                torch.nn.utils.clip_grad_norm_(group, hp.max_norm)
            for name in names:
                applied = joint if condition == "joint" else norms[name]
                sums["norm"][name] += norms[name]
                sums["scale"][name] += min(1.0, hp.max_norm / applied)
            joint_sum += joint
            for o in optimisers:
                o.step()
            for s in schedulers:
                s.step()
        for key in sums:
            for name in names:
                log[key][name][ep] = sums[key][name] / len(loader)
        log["joint_norm"][ep] = joint_sum / len(loader)
        mse = ", ".join(f"{n} {log['mse'][n][ep]:.2e}" for n in names)
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
    """Validation MSE per model, overall and per output step."""
    x_val, y_val = x_val.to(device), y_val.to(device)
    step_axes = (0,) + tuple(range(2, y_val.ndim))
    out = {}
    for name, model in models.items():
        err = (model(x_val[:, 0], num_time_steps=y_val.shape[1]) - y_val) ** 2
        out[name] = {
            "val_mse": float(err.mean()),
            "val_mse_per_step": err.mean(dim=step_axes).tolist(),
        }
    return out


def to_lists(log):
    return {
        k: ({n: v.tolist() for n, v in val.items()} if isinstance(val, dict) else val.tolist())
        for k, val in log.items()
    }


def save_models(models, norms, path):
    """State dicts (on CPU) and normalisation multipliers, to rebuild the models later."""
    torch.save(
        {
            "models": {
                n: {k: v.cpu() for k, v in m.state_dict().items()}
                for n, m in models.items()
            },
            "norms": norms.cpu(),
        },
        path,
    )
