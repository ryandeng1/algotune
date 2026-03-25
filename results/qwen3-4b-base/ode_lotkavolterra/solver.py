import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        params = problem["params"]
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]
        
        @njit
        def _lotka_volterra(t, y, alpha, beta, delta, gamma):
            x = y[0]
            y_pred = y[1]
            dx_dt = alpha * x - beta * x * y_pred
            dy_dt = delta * x * y_pred - gamma * y_pred
            return np.array([dx_dt, dy_dt])
        
        def lotka_volterra(t, y):
            return _lotka_volterra(t, y, alpha, beta, delta, gamma)
        
        sol = solve_ivp(
            lotka_volterra,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-10,
            atol=1e-10,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()
