import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True):
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]
        alpha, beta, delta, gamma = p["alpha"], p["beta"], p["delta"], p["gamma"]

        def lotka_volterra(t, y):
            x, y_val = y
            dx_dt = alpha * x - beta * x * y_val
            dy_dt = delta * x * y_val - gamma * y_val
            return np.array([dx_dt, dy_dt], dtype=float)

        rtol, atol = 1e-10, 1e-10
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            lotka_volterra,
            (t0, t1),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )