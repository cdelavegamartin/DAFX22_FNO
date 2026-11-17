"""Multi-seed figures of the training sweeps: training curves and validation MSE.

Training curves are drawn per epoch after a rolling median (training MSE spikes and the
per-epoch clip scale are noisy), as a centre line over seeds (geometric mean for the
log-scaled panels, arithmetic mean for the clip scale) and a band covering min-max.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MODEL_LABELS = {"gru": "FGRU", "rnn": "FRNN", "ref": "REF"}
# Slots 1-3 of the categorical palette of the dataviz reference
MODEL_COLORS = {"gru": "#eb6834", "rnn": "#1baf7a", "ref": "#2a78d6"}
SHARED_COLOR = "#5f5e5a"  # neutral: quantities shared by all models (joint clipping)
PANELS = (
    ("mse", "Training MSE", True),
    ("norm", "Gradient norm before clipping", True),
    ("scale", "Gradient scale applied by clipping", False),
)
SMOOTHING_EPOCHS = 25


def smooth(y, window=SMOOTHING_EPOCHS):
    """Centred rolling median, edges padded with the edge values."""
    if window <= 1 or len(y) < window:
        return np.asarray(y, dtype=float)
    padded = np.pad(np.asarray(y, dtype=float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def aggregate(curves, log_scale):
    """Centre (geometric mean if log_scale, else mean), min and max over seeds per epoch."""
    stacked = np.stack([smooth(c) for c in curves])
    centre = np.exp(np.log(stacked).mean(0)) if log_scale else stacked.mean(0)
    return centre, stacked.min(0), stacked.max(0)


def _band(ax, curves, color, log_scale):
    centre, lo, hi = aggregate(curves, log_scale)
    epochs = np.arange(1, len(centre) + 1)
    ax.fill_between(epochs, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(epochs, centre, color=color, lw=1.5)


def plot_training(rows, path, title):
    """Training curves, one row per condition and one column per quantity.

    rows: list of (row label, logs, joint), where logs holds one training log per seed
    (as written by dafx22_fno.training.train) and joint marks one clip shared by all
    models: its clip scale is drawn once, and its joint gradient norm is added.
    Rows share the y-axis of each column, so the conditions compare directly.
    """
    fig, axes = plt.subplots(
        len(rows), len(PANELS), figsize=(11, 2.6 * len(rows) + 0.6),
        sharex=True, sharey="col", layout="constrained", squeeze=False,
    )
    names = list(rows[0][1][0]["mse"])
    for (label, logs, joint), row_axes in zip(rows, axes):
        for ax, (key, _, log_scale) in zip(row_axes, PANELS):
            if key == "scale" and joint:
                _band(ax, [log[key][names[0]] for log in logs], SHARED_COLOR, log_scale)
            else:
                for name in names:
                    _band(ax, [log[key][name] for log in logs], MODEL_COLORS[name], log_scale)
            if key == "norm" and joint:
                _band(ax, [log["joint_norm"] for log in logs], SHARED_COLOR, log_scale)
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, color="#d8d7d2", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        row_axes[0].set_ylabel(f"{label}  (n = {len(logs)} seeds)", fontweight="bold")
    for ax, (_, column_title, _) in zip(axes[0], PANELS):
        ax.set_title(column_title, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    handles = [Line2D([], [], color=MODEL_COLORS[n], lw=1.5, label=MODEL_LABELS[n]) for n in names]
    handles.append(Line2D([], [], color=SHARED_COLOR, lw=1.5, label="all models (joint clip)"))
    handles.append(Patch(color=SHARED_COLOR, alpha=0.2, label="min-max over seeds"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle(
        f"{title}  (line: mean over seeds, geometric for log axes; "
        f"rolling median over {SMOOTHING_EPOCHS} epochs)",
        fontsize=10,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_validation(panels, path, xlabel):
    """Validation MSE per seed (small dots) and geometric mean over seeds (large marker).

    panels: {panel title: {x label: {condition: {model: [val MSE per seed]}}}}; conditions
    are told apart by marker fill (first filled, second hollow) and offset along x.
    """
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 3.4), layout="constrained", squeeze=False
    )
    conditions, names = [], []
    for ax, (panel_title, by_x) in zip(axes[0], panels.items()):
        xs = list(by_x)
        for i, x in enumerate(xs):
            for c, (condition, by_model) in enumerate(by_x[x].items()):
                if condition not in conditions:
                    conditions.append(condition)
                for m, (name, values) in enumerate(by_model.items()):
                    if name not in names:
                        names.append(name)
                    pos = i + (m - 1) * 0.22 + (conditions.index(condition) - 0.5) * 0.09
                    fill = MODEL_COLORS[name] if conditions.index(condition) == 0 else "none"
                    # Seeds beside their mean, so the mean marker does not hide them
                    ax.scatter(
                        [pos + 0.04] * len(values), values, s=10,
                        color=MODEL_COLORS[name], alpha=0.6, lw=0,
                    )
                    ax.scatter(
                        pos, np.exp(np.mean(np.log(values))), s=60, marker="o",
                        facecolors=fill, edgecolors=MODEL_COLORS[name], linewidths=1.5,
                    )
        ax.set_xticks(range(len(xs)), xs)
        ax.set(yscale="log", title=panel_title, xlabel=xlabel, ylabel="validation MSE")
        ax.grid(True, axis="y", color="#d8d7d2", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=MODEL_COLORS[n], label=MODEL_LABELS[n]) for n in names]
    for c, condition in enumerate(conditions):
        handles.append(
            Line2D([], [], ls="", marker="o", markersize=7, color=SHARED_COLOR,
                   markerfacecolor=SHARED_COLOR if c == 0 else "none",
                   label=f"{condition} (geometric mean)")
        )
    handles.append(Line2D([], [], ls="", marker="o", markersize=3, color=SHARED_COLOR, label="one seed"))
    # Models on the first row, markers on the second, so the legend fits one panel
    fig.legend(handles=handles, loc="outside lower center", ncol=len(names), frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
