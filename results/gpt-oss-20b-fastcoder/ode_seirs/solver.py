import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, Any], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]

        beta, sigma, gamma, omega = (
            p["beta"],
            p["sigma"],
            p["gamma"],
            p["omega"],
        )

        def seirs(t, y):
            S, E, I, R = y
            SIR = beta * S * I
            return [
                -SIR + omega * R,
                SIR - sigma * E,
                sigma * E - gamma * I,
                gamma * I - omega * R,
            ]

        rtol = atol = 1e-10
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )