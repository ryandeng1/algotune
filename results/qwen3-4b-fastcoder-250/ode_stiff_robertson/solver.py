import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0_arr = np.array(problem["y0"], dtype=float)
        k_arr = np.array(problem["k"], dtype=float)
        
        @njit(fastmath=True)
        def rober_numba(t, y, k):
            y1, y2, y3 = y
            k1, k2, k3 = k
            y2_sq = y2 * y2
            f0 = -k1 * y1 + k3 * y2 * y3
            f1 = k1 * y1 - k2 * y2_sq - k3 * y2 * y3
            f2 = k2 * y2_sq
            return np.array([f0, f1, f2], dtype=np.float64)
        
        sol = solve_ivp(
            fun=rober_numba,
            t_span=[t0, t1],
            y0=y0_arr,
            method="Radau",
            rtol=1e-11,
            atol=1e-9,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
