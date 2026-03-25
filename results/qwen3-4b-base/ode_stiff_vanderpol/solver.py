import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        mu = float(problem["mu"])
        y0 = np.array(problem["y0"])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        
        @numba.njit(nopython=True)
        def vdp_numba(t, y, mu):
            x = y[0]
            v = y[1]
            dx_dt = v
            dv_dt = mu * ((1 - x**2) * v - x)
            return np.array([dx_dt, dv_dt])
        
        sol = solve_ivp(
            vdp_numba,
            [t0, t1],
            y0,
            args=(mu,),
            method='BDF',
            rtol=1e-8,
            atol=1e-9,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
