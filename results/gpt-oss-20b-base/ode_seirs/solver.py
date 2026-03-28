import numpy as np
from scipy.integrate import solve_ivp

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = False) -> Any:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']

        # Extract parameters once for speed
        beta = params['beta']
        sigma = params['sigma']
        gamma = params['gamma']
        omega = params['omega']

        def seirs(t, y):
            S, E, I, R = y
            betaSI = beta * S * I
            dSdt = -betaSI + omega * R
            dEdt = betaSI - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            return np.array([dSdt, dEdt, dIdt, dRdt], dtype=np.float64)

        rtol = 1e-10
        atol = 1e-10
        method = 'RK45'

        t_eval = None if not debug else np.linspace(t0, t1, 1000)
        sol = solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol