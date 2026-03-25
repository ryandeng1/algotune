import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        params = problem["params"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        I = params["I"]
        
        @numba.jit(nopython=True)
        def rhs_numba(t, y, a, b, c, I):
            v = y[0]
            w = y[1]
            dv_dt = v - (v ** 3) / 3.0 - w + I
            dw_dt = a * (b * v - c * w)
            return np.array([dv_dt, dw_dt])
        
        def rhs_wrapper(t, y):
            return rhs_numba(t, y, a, b, c, I)
        
        sol = solve_ivp(
            rhs_wrapper,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
