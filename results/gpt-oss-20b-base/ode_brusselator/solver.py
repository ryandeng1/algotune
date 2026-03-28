import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1]
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.asarray(problem['y0'], dtype=float)
        t0, t1 = problem['t0'], problem['t1']
        A, B = problem['params']['A'], problem['params']['B']

        def brusselator(t, y):
            X, Y = y
            dX_dt = A + X * X * Y - (B + 1.0) * X
            dY_dt = B * X - X * X * Y
            return np.array([dX_dt, dY_dt], dtype=float)

        rtol = atol = 1e-8
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            brusselator,
            (t0, t1),
            y0,
            method='RK45',
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol