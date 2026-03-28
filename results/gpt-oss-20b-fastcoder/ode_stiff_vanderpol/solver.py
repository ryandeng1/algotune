from typing import Any, Dict, List
import math
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        result = self._solve(problem, debug=False)
        if result.success:
            return result.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {result.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        mu = float(problem["mu"])

        def vdp(t: float, y: np.ndarray) -> np.ndarray:
            x, v = y
            return np.array([v, mu * ((1 - x * x) * v - x)], dtype=float, order="F")

        method = "Radau"
        t_eval = np.linspace(t0, t1, 1_000, dtype=float) if debug else None
        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method=method,
            rtol=1e-8,
            atol=1e-9,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol