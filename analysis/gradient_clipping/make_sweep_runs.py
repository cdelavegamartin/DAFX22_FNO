"""Write sweep_runs.txt, the run list of clip_sweep.sbatch (one line of arguments per task).

Runs: 2 systems x seeds 0-2 x max norm {1, 10, 100, inf} x {joint, separate}, without
the runs already in results/ (seed 0, max norm 1 for both systems).

Usage: python analysis/gradient_clipping/make_sweep_runs.py
"""

from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEMS = ("string", "nonlinear_string")
SEEDS = range(3)
MAX_NORMS = (1.0, 10.0, 100.0, float("inf"))  # 1 is the paper's; inf disables clipping
CONDITIONS = ("joint", "separate")
DONE = {(system, 0, 1.0) for system in SYSTEMS}


def runs():
    for system, seed, max_norm, condition in product(SYSTEMS, SEEDS, MAX_NORMS, CONDITIONS):
        if (system, seed, max_norm) not in DONE:
            yield (
                f"--system {system} --seed {seed} --max-norm {max_norm:g} "
                f"--condition {condition}"
            )


def main():
    lines = list(runs())
    (HERE / "sweep_runs.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} runs written to {HERE / 'sweep_runs.txt'}")


if __name__ == "__main__":
    main()
