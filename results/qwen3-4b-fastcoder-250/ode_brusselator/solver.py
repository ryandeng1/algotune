import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def solve(self, problem, **kwargs):
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        params = problem["params"]
        A = params["A"]
        B = params["B"]
        
        @numba.jit(nopython=True)
        def brusselator_numba(t, y, A_val, B_val):
            X = y[0]
            Y = y[1]
            dX = A_val + X * X * Y - (B_val + 1) * X
            dY = B_val * X - X * X * Y
            return np.array([dX, dY])
        
        def ode_func(t, y):
            return brusselator_numba(t, y, A, B)
        
        sol = solve_ivp(
            ode_func,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=None
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
