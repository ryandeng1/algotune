import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        # Local copies to avoid attribute look‑ups in the ODE function
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k1, k2, k3 = tuple(problem["k"])

        # Preallocate result array to avoid repeated allocations
        res = np.empty(3)

        def rober(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3 = y
            # The `res` array is reused to keep the function lean
            res[0] = -k1 * y1 + k3 * y2 * y3
            res[1] = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
            res[2] = k2 * y2 * y2
            return res

        rtol, atol = 1e-11, 1e-9
        method = "Radau"

        if debug:
            # When debugging we want outputs at many points
            # The construction is cheap enough and keeps the call in a single line
            t_eval = np.clip(np.exp(np.linspace(np.log(1e-6), np.log(t1), 1000)), t0, t1)
        else:
            t_eval = None

        return solve_ivp(
            rober,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )