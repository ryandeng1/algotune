import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    # pylint: disable=missing-function-docstring
    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        a, b, c, I = (
            problem["params"]["a"],
            problem["params"]["b"],
            problem["params"]["c"],
            problem["params"]["I"],
        )

        # Limited overhead: pre-extracted constants, no dynamic look‑ups.
        def f(t, y):
            v, w = y
            dv = v - v * v * v / 3.0 - w + I
            dw = a * (b * v - c * w)
            return [dv, dw]

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            f,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=t_eval,
            dense_output=debug,
        )