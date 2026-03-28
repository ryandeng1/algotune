from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = False) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        def vdp(t: float, y: "np.ndarray[float, shape=(2,)]") -> "np.ndarray[float, shape=(2,)]":
            x, v = y
            return np.array([v, mu * ((1 - x ** 2) * v - x)], dtype=float)

        # Radau is well suited for stiff systems like the Van der Pol oscillator
        method = "Radau"
        rtol, atol = 1e-8, 1e-9

        # If debugging, we want intermediate points, otherwise just the end
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        return solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )