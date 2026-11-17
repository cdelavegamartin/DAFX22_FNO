"""Datasets of the DAFX22 paper (Parker et al., 2022), as generated at commit e08b4b0.

Each system simulates num_variations trajectories, half from the first kind of initial
condition (pluck or impulse at a random position) and half from random initial
conditions, in the same order of random draws as the original training scripts.
The original scripts did not seed numpy; here numpy is seeded with `seed`.

`simulate` returns one raw (unnormalised) float32 array of shape
(trajectories, time steps, x, [y,] channels); time step 0 is the model input and the
following steps are the target.
"""

import numpy as np

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


def simulate(system, num_variations=1024, seed=0):
    """Raw trajectories (trajectories, time steps, x, [y,] channels) as float32."""
    np.random.seed(seed)
    trajectories = _SIMULATORS[system](num_variations, SYSTEMS[system])
    # Move the time axis (second to last) to the front of each trajectory
    return np.stack([np.moveaxis(t, -2, 0) for t in trajectories]).astype(np.float32)


def split(data, val_split=0.1):
    """Training and validation trajectories; the last ceil(val_split * n) validate."""
    num_validation = int(np.ceil(val_split * data.shape[0]))
    return data[:-num_validation], data[-num_validation:]
