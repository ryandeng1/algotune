# solver.py
from __future__ import annotations
from typing import Any, Dict

import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """
    Optimised solver for the SEIRS model.  The integration routine is
    performed with `scipy.integrate.solve_ivp` using the high‑accuracy
    «RK45» scheme.  The right‑hand side is implemented in a style that
    minimises allocation and keeps the Python overhead to a minimum.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """Integrate the SEIRS system and return the final state."""
        result = self._solve(problem, debug=False)
        if result.success:
            return result.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {result.message}")

    def _solve(
        self,
        problem: Dict[str, np.ndarray | float],
        debug: bool = True,
    ) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Capture parameters locally so they are looked up only once
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]

        def seirs(t: float, y: np.ndarray) -> np.ndarray:
            S, E, I, R = y
            dSdt = -beta * S * I + omega * R
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            return np.array([dSdt, dEdt, dIdt, dRdt], dtype=y.dtype)

        rtol = atol = 1e-10
        method = "RK45"

        # When debugging we want dense output to inspect the trajectory; otherwise
        # we skip t_eval for speed.
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )