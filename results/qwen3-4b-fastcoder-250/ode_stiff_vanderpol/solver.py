import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        mu = float(problem['mu'])
        y0 = np.array(problem['y0'])
        t0 = float(problem['t0'])
        t1 = float(problem['t1'])
        
        @njit
        def vdp(t, y, mu_val):
            x = y[0]
            v = y[1]
            dx_dt = v
            dv_dt = mu_val * ((1 - x * x) * v - x)
            return np.array([dx_dt, dv_dt])
        
        def vdp_wrapper(t, y):
            return vdp(t, y, mu)
        
        vdp_wrapper_numba = njit(vdp_wrapper)
        
        sol = solve_ivp(
            vdp_wrapper_numba,
            [t0, t1],
            y0,
            method="Radau",
            rtol=1e-8,
            atol=1e-9,
            dense_output=False,
        )
        
        if sol.success:
            return [sol.y[0, -1], sol.y[1, -1]]
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
