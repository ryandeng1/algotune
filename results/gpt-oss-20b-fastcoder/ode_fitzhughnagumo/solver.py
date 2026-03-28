import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool) -> any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        a = problem["params"]["a"]
        b = problem["params"]["b"]
        c = problem["params"]["c"]
        I = problem["params"]["I"]

        # Local copy of the RHS to avoid attribute lookups per step
        def rhs(t, y):
            v, w = y
            dv_dt = v - v ** 3 / 3.0 - w + I
            dw_dt = a * (b * v - c * w)
            return np.array([dv_dt, dw_dt], dtype=np.float64)

        rtol, atol = 1e-8, 1e-8
        if debug:
            t_eval = np.linspace(t0, t1, 1000, dtype=np.float64)
            sol = solve_ivp(
                rhs,
                (t0, t1),
                y0,
                method="RK45",
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                dense_output=True,
            )
        else:
            sol = solve_ivp(
                rhs,
                (t0, t1),
                y0,
                method="RK45",
                rtol=rtol,
                atol=atol,
                dense_output=False,
            )
        return sol