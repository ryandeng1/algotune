from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        beta, sigma, gamma, omega = (
            problem['params']['beta'],
            problem['params']['sigma'],
            problem['params']['gamma'],
            problem['params']['omega'],
        )

        def seirs(_: float, y: np.ndarray):
            S, E, I, R = y
            dS = -beta * S * I + omega * R
            dE = beta * S * I - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I - omega * R
            return np.array([dS, dE, dI, dR], dtype=np.float64)

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-10,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol