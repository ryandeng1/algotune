import numpy as np
from diffrax import solve, ODETerm

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        t0 = problem['t0']
        t1 = problem['t1']
        y0 = np.array(problem['y0'])
        params = problem['params']
        A = params['A']
        B = params['B']
        
        def brusselator(t, y):
            X, Y = y
            dX_dt = A + X**2 * Y - (B + 1) * X
            dY_dt = B * X - X**2 * Y
            return np.array([dX_dt, dY_dt])
        
        term = ODETerm(brusselator)
        sol = solve(
            term,
            t0=t0,
            t1=t1,
            y0=y0,
            rtol=1e-8,
            atol=1e-8,
            saveat=[t1],
            method='DOP853',
        )
        return sol.y[0]
