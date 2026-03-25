import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    def solve(self, problem, **kwargs):
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        params = problem["params"]
        
        a = params["a"]
        b = params["b"]
        c = params["c"]
        I = params["I"]
        
        @njit
        def fitzhugh_nagumo(t, y):
            v = y[0]
            w = y[1]
            dv = v - (v ** 3) / 3.0 - w + I
            dw = a * (b * v - c * w)
            return (dv, dw)
        
        sol = solve_ivp(
            fitzhugh_nagumo,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            t_eval=None,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
