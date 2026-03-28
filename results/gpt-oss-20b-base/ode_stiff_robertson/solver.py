from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(
        self, problem: Dict[str, np.ndarray | float], debug: bool = True
    ) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k1, k2, k3 = problem["k"]

        def rober(t, y):
            y1, y2, y3 = y
            f0 = -k1 * y1 + k3 * y2 * y3
            f1 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
            f2 = k2 * y2 * y2
            return [f0, f1, f2]  # list is cheaper than 1‑D array allocation

        t_eval = (
            np.clip(
                np.exp(
                    np.linspace(np.log(1e-6), np.log(t1), 1000)
                ),
                t0,
                t1,
            )
            if debug
            else None
        )
        return solve_ivp(
            rober,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-11,
            atol=1e-9,
            t_eval=t_eval,
            dense_output=debug,
        )