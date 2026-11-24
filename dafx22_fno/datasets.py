"""Datasets of the DAFX22 paper (Parker et al., 2022), as generated at commit e08b4b0.

Each system simulates num_variations trajectories, half from the first kind of initial
condition (pluck or impulse at a random position) and half from random initial
conditions, in the same order of random draws as the original training scripts.
The original scripts did not seed numpy; here numpy is seeded with `seed`.

`simulate` returns one raw (unnormalised) float32 array of shape
(trajectories, time steps, x, [y,] channels); time step 0 is the model input and the
following steps are the target.
"""

import copy

import numpy as np
import torch

from dafx22_fno.generators.string_solver import StringSolver
from dafx22_fno.generators.tension_modulated_string_solver import (
    TensionModulatedStringSolver,
)
from dafx22_fno.generators.wave_solver_2d import WaveSolver2D

# Solver settings of train_1d_string.py, train_1d_nonlinear_string.py and
# train_2d_wave.py at commit e08b4b0
SYSTEMS = {
    "string": {
        "solver": dict(dur=0.0025, Fs=48000, delta_x=1e-2, d1=1e-1),
        "channels": ["deflection", "velocity"],
    },
    "nonlinear_string": {
        "solver": dict(dur=0.0025, Fs=48000, delta_x=1e-2, d1=1e-2),
        "max_pluck_deflection": 1e-3,
        "channels": ["deflection", "velocity"],
    },
    "wave2d": {
        "solver": dict(
            dur=0.0007, Fs=48000, lx=1, ly=0.95, spatial_delta=1 / 40, modes=25
        ),
        "channels": ["pressure", "velocity_x", "velocity_y"],
    },
}


def _string(num, settings):
    solver = StringSolver(**settings["solver"])
    for i in range(num):
        if i < num // 2:
            fe_x = solver.create_pluck(np.random.rand(1))
        else:
            fe_x = solver.create_random_initial()
        yield np.stack(solver.solve(fe_x), axis=-1)  # (x, t, channel)


def _nonlinear_string(num, settings):
    solver = TensionModulatedStringSolver(**settings["solver"])
    for i in range(num):
        hi = np.random.rand(1) * settings["max_pluck_deflection"]
        if i < num // 2:
            fe_x = solver.create_pluck(np.random.rand(1), hi)
        else:
            fe_x = solver.create_random_initial(hi)
        yield np.stack(solver.solve(fe_x), axis=-1)  # (x, t, channel)


def _wave2d(num, settings):
    solver = WaveSolver2D(**settings["solver"])
    for i in range(num):
        if i < num // 2:
            fe_x = solver.create_impulse(np.random.rand(1), np.random.rand(1))
        else:
            fe_x = solver.create_random_initial()
        _, _, pressure, velocity_x, velocity_y = solver.solve(fe_x)
        yield np.stack([pressure, velocity_x, velocity_y], axis=-1)  # (x, y, t, ch)


_SIMULATORS = {"string": _string, "nonlinear_string": _nonlinear_string, "wave2d": _wave2d}


def settings(system, steps=None):
    """Settings of a system; steps sets the number of time steps (input included).

    The solvers use round(dur * Fs) time steps, so dur = steps / Fs gives exactly
    `steps`; without it the duration of the paper is kept.
    """
    result = copy.deepcopy(SYSTEMS[system])
    if steps is not None:
        result["solver"]["dur"] = steps / result["solver"]["Fs"]
    return result


def simulate(system, num_variations=1024, seed=0, steps=None):
    """Raw trajectories (trajectories, time steps, x, [y,] channels) as float32."""
    np.random.seed(seed)
    trajectories = _SIMULATORS[system](num_variations, settings(system, steps))
    # Move the time axis (second to last) to the front of each trajectory
    return np.stack([np.moveaxis(t, -2, 0) for t in trajectories]).astype(np.float32)


def split(data, val_split=0.1):
    """Training and validation trajectories; the last ceil(val_split * n) validate."""
    num_validation = int(np.ceil(val_split * data.shape[0]))
    return data[:-num_validation], data[-num_validation:]


def load_normalised(path, val_split=0.1, train_steps=None):
    """Normalised (x_train, y_train, x_val, y_val) and norms from a generate_dataset.py array.

    As in e08b4b0, each channel is scaled by the std of the output steps of all
    trajectories, and the last ceil(val_split * n) trajectories validate.
    With `train_steps`, training targets and the std use only the first train_steps
    output steps, and validation keeps all steps (for extrapolation).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; create it with generate_dataset.py (see its --help)"
        )
    data = torch.from_numpy(np.load(path))
    inputs, outputs = data[:, :1], data[:, 1:]
    steps = train_steps or outputs.shape[1]
    norms = 1 / torch.std(outputs[:, :steps], dim=tuple(range(outputs.ndim - 1)))
    inputs, outputs = inputs * norms, outputs * norms
    (x_train, x_val), (y_train, y_val) = split(inputs, val_split), split(outputs, val_split)
    return x_train, y_train[:, :steps], x_val, y_val, norms
