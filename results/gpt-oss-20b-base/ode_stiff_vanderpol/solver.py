from __future__ import annotations
import numpy as np
from typing import Any, Dict, List


class Solver:
    """
    A very lightweight and fast solver that evaluates a linear time‑invariant
    system of the form

        dy/dt = A @ y + b

    with an explicit Euler integrator.  The solver is intentionally simple
    so that it does not pull in heavyweight packages (e.g. SciPy) and
    therefore offers excellent performance on typical I/O‑bound workloads.

    The expected input ``problem`` is a mapping that contains at least:

    * ``A``   – the system matrix (`np.ndarray`, shape ``(n, n)``).
    * ``b``   – a constant forcing vector (`np.ndarray`, shape ``(n,)``).
    * ``y0``  – the initial state vector (`np.ndarray`, shape ``(n,)``).
    * ``t``   – an array of monotonically increasing time points
                (`np.ndarray`, shape ``(m,)``).  Time steps may be
                non‑uniform.

    The output is a mapping with a single key ``"final_state"`` that
    contains the state vector at the last time point as a plain Python
    list of floats.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        A = problem["A"]
        b = problem["b"]
        y = problem["y0"].astype(float, copy=True)
        t = problem["t"]

        dt = np.diff(t)
        n = y.size

        # Explicit Euler integration
        for i, step in enumerate(dt):
            y += step * (A @ y + b)

        return {"final_state": y.tolist()}