"""Compare the spectral truncation of FourierConv{1,2}d (FGRU/FRNN) and SpectralConv{1,2}d (REF).

Investigates cdelavegamartin/phd-thesis#85: FourierConv2d keeps only out_ft[:, :, :Nx, :Ny]
of an rfft2 over a (2Nx, 2Ny) zero-padded grid, which discards the negative kx half of the
half-spectrum, i.e. the (kx<0, ky>0) / (kx>0, ky<0) quadrant pair.

The script imports the layers from this repository unchanged and runs:
  1. exact equivalence: FourierConvNd == SpectralConvNd restricted to the positive-kx block,
  2. retained-bin masks of each layer in the (kx, ky) plane,
  3. identity-weight passband maps for 2D plane waves (and the 1D sanity check),
  4. energy retained for clamped membrane modes sin(m pi x / Lx) sin(n pi y / Ly),
  5. orientation test cos(a x + b y) vs cos(a x - b y) for random and trained weights.

Usage:
  uv run python analysis/spectral_truncation/compare_spectral_layers.py [--runs-dir OUTPUT_DIR]
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dafx22_fno.modules.fno_ref import SpectralConv1d, SpectralConv2d  # noqa: E402
from dafx22_fno.modules.layers import FourierConv1d, FourierConv2d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_spectrum import FourierConv2dFullSpectrum  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

# Grid of train_2d_wave.py: num_points = 40, lx = 1, ly = 0.95, spatial_delta = 1 / 40.
NX, NY = 40, 38
LX, LY = 1.0, 0.95
# Grid of train_1d_string.py: l = 1, delta_x = 1e-2.
N1D = 100
# FNO_Markov_{1,2}d hyper-parameters.
REF_PAD_2D, REF_PAD_1D, REF_MODES = 9, 2, 20
WIDTH = 8

# Categorical slots (validated reference palette) and plot style.
C_FOURIER, C_SPECTRAL, C_FIXED = "#eb6834", "#2a78d6", "#1baf7a"
CMAP_SEQ, CMAP_DIV = "Blues", "RdBu_r"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)

torch.manual_seed(0)
CFLOAT = torch.complex64  # float32 throughout, as in training


# ---------------------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------------------
def to_real_view(z):
    return torch.view_as_real(z.to(CFLOAT)).clone()


def set_param(param, value):
    with torch.no_grad():
        param.data = value.to(param.dtype).clone()


def set_identity(*params):
    """Set complex weight tensors (stored as real views) to 1 + 0j, single channel."""
    for p in params:
        set_param(p, to_real_view(torch.ones(p.shape[:-1], dtype=CFLOAT)))


def build_layers_2d(nx=NX, ny=NY, identity=True, channels=1):
    """Return {name: callable (B, C, nx, ny) -> (B, C, nx, ny)} for the three 2D variants."""
    fourier = FourierConv2d(channels, channels, nx, ny, bias=False)
    spectral = SpectralConv2d(channels, channels, REF_MODES, REF_MODES)
    fixed = FourierConv2dFullSpectrum(channels, channels, nx, ny, bias=False)
    if identity:
        set_identity(fourier.weights, spectral.weights1, spectral.weights2)
        set_identity(fixed.weights, fixed.weights_neg)
    return {
        "FourierConv2d (FGRU/FRNN)": fourier,
        "SpectralConv2d (REF, pad 9, 20 modes)": ref_path_2d(spectral),
        "FourierConv2d + negative-kx block (fix)": fixed,
    }


def ref_path_2d(layer, pad=REF_PAD_2D):
    """Spectral path of FNO_Markov_2d.cell: pad, SpectralConv2d, crop."""

    def apply(x):
        return layer(F.pad(x, [0, pad, 0, pad]))[..., :-pad, :-pad]

    apply.layer = layer
    return apply


def ref_path_1d(layer, pad=REF_PAD_1D):
    def apply(x):
        return layer(F.pad(x, [0, pad]))[..., :-pad]

    return apply


LAYER_COLORS = [C_FOURIER, C_SPECTRAL, C_FIXED]


def wrap(name):
    """Two-line panel title: layer name, then the parenthesised context."""
    return name.replace(" (", "\n(")


# ---------------------------------------------------------------------------------------
# Signals and metrics
# ---------------------------------------------------------------------------------------
def grid_2d(nx=NX, ny=NY, lx=LX, ly=LY):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    return np.meshgrid(x, y, indexing="ij")


def dft_wave(p, q, phase=0.0, nx=NX, ny=NY):
    """cos(2 pi (p i / nx + q j / ny) + phase) on the index grid."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.cos(2 * np.pi * (p * i / nx + q * j / ny) + phase)


def clamped_mode(m, n):
    x, y = grid_2d()
    return np.sin(m * np.pi * x / LX) * np.sin(n * np.pi * y / LY)


def as_batch(fields):
    """List of 2D (or 1D) arrays -> tensor (B, 1, ...)."""
    return torch.tensor(np.stack(fields), dtype=torch.float32)[:, None]


def energy(t):
    return float((t**2).sum())


@torch.no_grad()
def energy_ratio(layer, fields):
    """Output energy over input energy, summed over a list of input fields."""
    x = as_batch(fields)
    return energy(layer(x)) / energy(x)


@torch.no_grad()
def linear_response(layer, x):
    """layer(x) - layer(0): removes any input-independent bias term."""
    return layer(x) - layer(torch.zeros_like(x))


# ---------------------------------------------------------------------------------------
# 1. Exact equivalence
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def check_equivalence():
    """FourierConvNd == SpectralConvNd with weights on the positive-kx block only."""
    c = 3
    res = {}

    # 2D: SpectralConv2d with modes (NX, NY) on the same (2NX, 2NY) zero-padded grid.
    fourier = FourierConv2d(c, c, NX, NY, bias=False)
    spectral = SpectralConv2d(c, c, NX, NY)
    set_param(spectral.weights1, fourier.weights.data)
    set_param(spectral.weights2, torch.zeros_like(spectral.weights2))
    x = torch.randn(4, c, NX, NY)

    def spectral_on_fourier_grid(inp):
        return spectral(F.pad(inp, [0, NY, 0, NX]))[..., :NX, :NY]

    y_f = fourier(x)
    y_s = spectral_on_fourier_grid(x)
    res["2d_max_abs_diff_weights2_zero"] = float((y_f - y_s).abs().max())
    res["2d_output_rms"] = float(y_f.pow(2).mean().sqrt())
    set_param(spectral.weights2, fourier.weights.data)
    res["2d_max_abs_diff_weights2_equal_weights1"] = float(
        (y_f - spectral_on_fourier_grid(x)).abs().max()
    )

    # 1D: SpectralConv1d with modes N on the same 2N zero-padded grid.
    fourier1 = FourierConv1d(c, c, N1D, bias=False)
    spectral1 = SpectralConv1d(c, c, N1D)
    set_param(spectral1.weights1, fourier1.weights.data)
    x1 = torch.randn(4, c, N1D)
    y1 = spectral1(F.pad(x1, [0, N1D]))[..., :N1D]
    res["1d_max_abs_diff"] = float((fourier1(x1) - y1).abs().max())
    res["1d_output_rms"] = float(fourier1(x1).pow(2).mean().sqrt())
    return res


# ---------------------------------------------------------------------------------------
# 2. Retained-bin masks
# ---------------------------------------------------------------------------------------
@contextlib.contextmanager
def capture_irfft2():
    """Record the spectrum passed to torch.fft.irfft2 inside a layer's forward."""
    captured = []
    original = torch.fft.irfft2

    def wrapper(inp, *args, **kwargs):
        captured.append((inp.detach().clone(), kwargs.get("s")))
        return original(inp, *args, **kwargs)

    torch.fft.irfft2 = wrapper
    try:
        yield captured
    finally:
        torch.fft.irfft2 = original


@torch.no_grad()
def full_plane_mask(layer):
    """Boolean mask of non-zero output bins in the full fftshifted (kx, ky) plane."""
    with capture_irfft2() as cap:
        layer(torch.randn(1, 1, NX, NY))
    if not cap:  # ref_path_2d wraps the layer in a closure
        raise RuntimeError("irfft2 was not called")
    spec, size = cap[0]
    half = spec[0, 0].abs() > 0  # (H, W//2 + 1), ky >= 0
    h = half.shape[0]
    w = size[-1] if size is not None else 2 * (half.shape[1] - 1)
    full = torch.zeros(h, w, dtype=torch.bool)
    full[:, : half.shape[1]] = half
    # Hermitian partner of bin (kx, ky) is (-kx, -ky): fill ky < 0 from ky > 0.
    kx = torch.arange(h)
    for ky in range(1, half.shape[1]):
        if ky < w - ky:
            full[(-kx) % h, w - ky] |= half[kx, ky]
    freqs_x = np.fft.fftshift(np.fft.fftfreq(h))
    freqs_y = np.fft.fftshift(np.fft.fftfreq(w))
    return np.fft.fftshift(full.numpy()), freqs_x, freqs_y


def plot_masks(layers, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (name, layer), color in zip(axes, layers.items(), LAYER_COLORS):
        mask, fx, fy = full_plane_mask(layer)
        cmap = matplotlib.colors.ListedColormap(["#f2f2f0", color])
        ax.pcolormesh(fy, fx, mask, cmap=cmap, shading="nearest", rasterized=True)
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.axvline(0, color="#52514e", lw=0.6)
        ax.set_title(wrap(name), fontsize=8)
        ax.set_xlabel(r"$k_y$ (cycles/sample)")
        ax.set_aspect("equal")
        frac = mask.mean()
        ax.text(
            0.02, 0.02, f"{100 * frac:.0f}% of bins", transform=ax.transAxes, fontsize=7
        )
    axes[0].set_ylabel(r"$k_x$ (cycles/sample)")
    fig.suptitle(
        "Output bins each layer can populate (full plane, Hermitian extension)",
        fontsize=9,
    )
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
# 3. Passband maps
# ---------------------------------------------------------------------------------------
def passband_2d(layer):
    ps = np.arange(-NX // 2, NX // 2 + 1)
    qs = np.arange(0, NY // 2 + 1)
    out = np.zeros((len(ps), len(qs)))
    for a, p in enumerate(ps):
        for b, q in enumerate(qs):
            fields = [dft_wave(p, q), dft_wave(p, q, np.pi / 2)]
            out[a, b] = energy_ratio(layer, fields)
    return ps, qs, out


def plot_passbands(layers, path):
    results = {name: passband_2d(layer) for name, layer in layers.items()}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    for ax, (name, (ps, qs, r)) in zip(axes, results.items()):
        im = ax.pcolormesh(qs, ps, r, cmap=CMAP_SEQ, vmin=0, vmax=1, shading="nearest")
        ax.set_title(wrap(name), fontsize=8)
        ax.set_xlabel(r"$q$ (cycles over $N_y$)")
        ax.axhline(0, color="#52514e", lw=0.6)
    axes[0].set_ylabel(r"$p$ (cycles over $N_x$)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label(r"$E_\mathrm{out}/E_\mathrm{in}$ (identity weights)")
    fig.suptitle(
        r"Plane-wave energy retained, $\cos(2\pi(pi/N_x + qj/N_y))$ on the 40 x 38 grid",
        fontsize=9,
    )
    fig.savefig(path)
    plt.close(fig)
    summary = {}
    for name, (ps, qs, r) in results.items():
        pos = r[(ps > 0)][:, qs > 0]
        neg = r[(ps < 0)][:, qs > 0]
        summary[name] = {
            "mean_ratio_p>0_q>0": float(pos.mean()),
            "mean_ratio_p<0_q>0": float(neg.mean()),
            "mean_ratio_q=0_p!=0": float(r[ps != 0][:, 0].mean()),
            "mean_ratio_p=0_q>0": float(r[ps == 0][0, qs > 0].mean()),
        }
    return summary


@torch.no_grad()
def passband_1d():
    fourier = FourierConv1d(1, 1, N1D, bias=False)
    spectral = SpectralConv1d(1, 1, REF_MODES)
    spectral_full = SpectralConv1d(1, 1, N1D)
    set_identity(fourier.weights, spectral.weights1, spectral_full.weights1)
    layers = {
        "FourierConv1d (FGRU/FRNN)": fourier,
        "SpectralConv1d (REF, pad 2, 20 modes)": ref_path_1d(spectral),
        "SpectralConv1d on FourierConv1d grid (N modes, pad N)": ref_path_1d(
            spectral_full, N1D
        ),
    }
    ks = np.arange(0, N1D // 2 + 1)
    i = np.arange(N1D)
    out = {}
    for name, layer in layers.items():
        out[name] = np.array(
            [
                energy_ratio(
                    layer,
                    [np.cos(2 * np.pi * k * i / N1D), np.sin(2 * np.pi * k * i / N1D)],
                )
                if k not in (0, N1D // 2)
                else energy_ratio(layer, [np.cos(2 * np.pi * k * i / N1D)])
                for k in ks
            ]
        )
    return ks, out


def plot_passband_1d(path):
    ks, out = passband_1d()
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    styles = ["-", "-", "--"]
    for (name, r), color, ls in zip(out.items(), LAYER_COLORS, styles):
        ax.plot(ks, r, ls, color=color, lw=2, label=name)
    ax.set_xlabel(r"$k$ (cycles over $N = 100$)")
    ax.set_ylabel(r"$E_\mathrm{out}/E_\mathrm{in}$")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("1D sanity check: identity weights, cosine/sine pair per $k$")
    fig.savefig(path)
    plt.close(fig)
    fourier_name = next(iter(out))
    return {
        name: {
            "min_ratio_k<N/2": float(r[1:-1].min()),
            "max_ratio_k<N/2": float(r[1:-1].max()),
        }
        for name, r in out.items()
    } | {"fourier_ratio_at_nyquist": float(out[fourier_name][-1])}


# ---------------------------------------------------------------------------------------
# 4. Physical examples and clamped modes
# ---------------------------------------------------------------------------------------
def issue_examples(layers):
    x, y = grid_2d()
    fields = {
        "cos(20x + 20y)": np.cos(20 * x + 20 * y),
        "cos(20x - 20y)": np.cos(20 * x - 20 * y),
        "sin(2 pi x) sin(3 pi y / 0.95)": clamped_mode(2, 3),
    }
    table = {
        f: {name: energy_ratio(layer, [u]) for name, layer in layers.items()}
        for f, u in fields.items()
    }
    return fields, table


@torch.no_grad()
def plot_examples(layers, fields, path):
    n_rows, n_cols = len(fields), 1 + len(layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.9 * n_rows))
    for r, (fname, u) in enumerate(fields.items()):
        x_in = as_batch([u])
        outs = [x_in[0, 0]] + [layer(x_in)[0, 0] for layer in layers.values()]
        titles = ["input"] + list(layers)
        for c, (field, title) in enumerate(zip(outs, titles)):
            ax = axes[r, c]
            ax.imshow(field.numpy().T, origin="lower", cmap=CMAP_DIV, vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(wrap(title), fontsize=7)
            if c == 0:
                ax.set_ylabel(fname, fontsize=7)
            if c > 0:
                ratio = energy(field) / energy(x_in)
                ax.text(
                    0.03,
                    0.04,
                    f"{100 * ratio:.1f}%",
                    transform=ax.transAxes,
                    fontsize=7,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1),
                )
    fig.suptitle(
        "Identity-weight outputs on the 40 x 38 grid (x horizontal, y vertical); "
        "% = energy retained",
        fontsize=8,
    )
    fig.savefig(path)
    plt.close(fig)


def clamped_mode_table(layers, m_max=15, n_max=15):
    out = {}
    for name, layer in layers.items():
        r = np.zeros((m_max, n_max))
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                r[m - 1, n - 1] = energy_ratio(layer, [clamped_mode(m, n)])
        out[name] = r
    return out


def plot_clamped(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, (name, r) in zip(axes, table.items()):
        m_max, n_max = r.shape
        im = ax.pcolormesh(
            np.arange(1, n_max + 1),
            np.arange(1, m_max + 1),
            r,
            cmap=CMAP_SEQ,
            vmin=0,
            vmax=1,
            shading="nearest",
        )
        ax.set_title(wrap(name), fontsize=8)
        ax.set_xlabel("$n$")
        ax.set_aspect("equal")
        ax.text(
            0.03,
            0.03,
            f"median {100 * np.median(r):.0f}%",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1),
        )
    axes[0].set_ylabel("$m$")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label(r"$E_\mathrm{out}/E_\mathrm{in}$")
    fig.suptitle(
        r"Clamped modes $\sin(m\pi x/L_x)\,\sin(n\pi y/L_y)$, identity weights",
        fontsize=9,
    )
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
# 5. Orientation test with random and trained weights
# ---------------------------------------------------------------------------------------
ORIENT_PQ = [(p, q) for p in (1, 2, 3, 5, 8) for q in (1, 2, 3, 5, 8)]


def random_orientation(n_draws=100):
    out = {"FourierConv2d": [], "SpectralConv2d": []}
    for _ in range(n_draws):
        f = FourierConv2d(WIDTH, WIDTH, NX, NY, bias=True)
        s = SpectralConv2d(WIDTH, WIDTH, REF_MODES, REF_MODES)
        out["FourierConv2d"].append(orientation_ratio(f, NX, NY))
        out["SpectralConv2d"].append(orientation_ratio(ref_path_2d(s), NX, NY, WIDTH))
    return out


def trained_orientation(runs_dir):
    out = {"FourierConv2d": [], "SpectralConv2d": []}
    runs, skipped = [], set()
    if runs_dir is None or not runs_dir.is_dir():
        return out, runs, []
    for run in sorted(runs_dir.glob("2d_wave_*")):
        files = {k: run / f"model_{k}.pt" for k in ("gru", "rnn", "ref")}
        if not all(f.exists() for f in files.values()):
            continue
        runs.append(run.name)
        for kind, f in files.items():
            model = torch.load(f, map_location="cpu", weights_only=False)
            model = getattr(model, "module", model)
            for mod in model.modules():
                if isinstance(mod, FourierConv2d):
                    out["FourierConv2d"].append(
                        orientation_ratio(mod, mod.size_x, mod.size_y)
                    )
                elif isinstance(mod, SpectralConv2d):
                    nx, ny = model.spatial_size_x, model.spatial_size_y
                    if not ref_fits_grid(mod, nx + model.padding, ny + model.padding):
                        skipped.add(f"{run.name} ({kind}, {nx}x{ny} grid)")
                        continue
                    out["SpectralConv2d"].append(
                        orientation_ratio(
                            ref_path_2d(mod, model.padding), nx, ny, mod.in_channels
                        )
                    )
    return out, runs, sorted(skipped)


def ref_fits_grid(layer, h, w):
    """The current SpectralConv2d needs 2 * modes1 <= h and modes2 <= w // 2 + 1."""
    return 2 * layer.modes1 <= h and layer.modes2 <= w // 2 + 1


@torch.no_grad()
def orientation_ratio(layer, nx, ny, channels=None):
    """Median over (p, q) of E(cos(-p, q)) / E(cos(p, q)) for the linear response.

    Each input channel receives the wave in turn (batch of one-hot channels).
    """
    channels = channels or layer.in_channels
    ratios = []
    eye = torch.eye(channels)[:, :, None, None]
    for p, q in ORIENT_PQ:
        e = {}
        for sign in (1, -1):
            u = torch.tensor(dft_wave(sign * p, q, nx=nx, ny=ny), dtype=torch.float32)
            e[sign] = energy(linear_response(layer, eye * u))
        ratios.append(e[-1] / e[1])
    return float(np.median(ratios))


def plot_orientation(random_res, trained_res, path):
    groups = [
        ("FourierConv2d\nrandom init", random_res["FourierConv2d"], C_FOURIER),
        ("FourierConv2d\ntrained FGRU/FRNN", trained_res["FourierConv2d"], C_FOURIER),
        ("SpectralConv2d\nrandom init", random_res["SpectralConv2d"], C_SPECTRAL),
        ("SpectralConv2d\ntrained REF", trained_res["SpectralConv2d"], C_SPECTRAL),
    ]
    groups = [g for g in groups if len(g[1])]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    rng = np.random.default_rng(0)
    for i, (label, vals, color) in enumerate(groups):
        vals = np.asarray(vals)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(
            i + jitter, vals, s=12, color=color, alpha=0.7, lw=0.5, edgecolor="white"
        )
        ax.hlines(np.median(vals), i - 0.28, i + 0.28, color="#0b0b0b", lw=1.5)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7)
    ax.set_yscale("log")
    ax.axhline(1, color="#52514e", lw=0.6, ls=":")
    ax.set_ylabel(r"$E[\cos(-p,q)]\,/\,E[\cos(p,q)]$")
    ax.set_title(
        "Orientation asymmetry of the linear response (median over 25 (p, q) pairs)"
    )
    fig.savefig(path)
    plt.close(fig)


def summarise(vals):
    v = np.asarray(vals)
    if not len(v):
        return {}
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


# ---------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "output",
        help="directory with 2d_wave_* training runs (model_{gru,rnn,ref}.pt)",
    )
    args = parser.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    results = {}

    results["equivalence"] = check_equivalence()

    layers = build_layers_2d()
    plot_masks(layers, FIG_DIR / "fig1_retained_bins.png")
    results["passband_2d"] = plot_passbands(layers, FIG_DIR / "fig2_passband_2d.png")

    fields, results["issue_examples"] = issue_examples(layers)
    plot_examples(layers, fields, FIG_DIR / "fig3_examples.png")

    clamped = clamped_mode_table(layers)
    plot_clamped(clamped, FIG_DIR / "fig4_clamped_modes.png")
    results["clamped_modes_1to15"] = {
        name: {
            "median": float(np.median(r)),
            "min": float(r.min()),
            "max": float(r.max()),
        }
        for name, r in clamped.items()
    }

    rand = random_orientation()
    trained, runs, skipped = trained_orientation(args.runs_dir)
    plot_orientation(rand, trained, FIG_DIR / "fig5_orientation.png")
    results["orientation_random_init"] = {k: summarise(v) for k, v in rand.items()}
    results["orientation_trained"] = {k: summarise(v) for k, v in trained.items()}
    results["trained_runs"] = runs
    results["trained_ref_skipped"] = skipped

    results["passband_1d"] = plot_passband_1d(FIG_DIR / "fig6_passband_1d.png")

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
