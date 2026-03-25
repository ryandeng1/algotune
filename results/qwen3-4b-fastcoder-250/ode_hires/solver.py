import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        t0 = problem['t0']
        t1 = problem['t1']
        y0 = np.array(problem['y0'])
        constants = problem['constants']
        constants_tuple = tuple(constants)
        
        @numba.njit
        def hires_ode(t, y):
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants_tuple
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            c10_y6_y8 = c10 * y6 * y8
            f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            f2 = c1 * y1 - c5 * y2
            f3 = -c6 * y3 + c2 * y4 + c7 * y5
            f4 = c3 * y2 + c1 * y3 - c8 * y4
            f5 = -c9 * y5 + c2 * y6 + c2 * y7
            f6 = -c10_y6_y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            f7 = c10_y6_y8 - c12 * y7
            f8 = -c10_y6_y8 + c12 * y7
            return np.array([f1, f2, f3, f4, f5, f6, f7, f8])
        
        sol = solve_ivp(
            hires_ode,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-9,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
