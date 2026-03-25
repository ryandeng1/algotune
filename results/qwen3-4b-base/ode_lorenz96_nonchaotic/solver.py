import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        F = float(problem["F"])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.array(problem["y0"])
        
        def lorenz96(t, x):
            N = len(x)
            ip1 = np.roll(np.arange(N), -1)
            im1 = np.roll(np.arange(N), 1)
            im2 = np.roll(np.arange(N), 2)
            return (x[ip1] - x[im2]) * x[im1] - x + F
        
        rtol = 1e-8
        atol = 1e-8
        method = 'DOP853'
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
