from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

class Solver:

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, list[float]]:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']
        A = float(params['A'])
        B = float(params['B'])

        # JIT‑compiled RHS of the Brusselator ODE
        @jit(nopython=True)
        def rhs(t, y):
            X, Y = y
            dX_dt = A + X*X*Y - (B + 1.0)*X
            dY_dt = B*X - X*X*Y
            return np.array([dX_dt, dY_dt], dtype=np.float64)

        # Solve without evaluating all intermediate points
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,      # no dense output, faster
            t_eval=None              # only final state needed
        )
        if not sol.success:
            raise RuntimeError(f'Solver failed: {sol.message}')
        return {'X': sol.y[0, -1].tolist(), 'Y': sol.y[1, -1].tolist()}