from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = True) -> Any:
        # Extract initial condition and bounds
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        params = problem["params"]

        # Pull parameters into local vars to avoid repeated dict lookups
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])

        def fitzhugh_nagumo(t: float, y: np.ndarray) -> np.ndarray:
            v, w = y
            dv_dt = v - v**3 / 3.0 - w + I
            dw_dt = a * (b * v - c * w)
            return np.array([dv_dt, dw_dt], dtype=float)

        # Configure solver options
        rtol = 1e-8
        atol = 1e-8
        method = "RK45"

        # No t_eval when not debugging (just final state)
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            fitzhugh_nagumo,
            (t0, t1),
            y0,
            method=method,
            t_eval=t_eval,
            dense_output=debug,
            rtol=rtol,
            atol=atol,
        )
        return sol