import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

class Solver:
    """
    Fast SEIRS solver.
    """
    @staticmethod
    @nb.njit
    def _seirs(t, y, beta, sigma, gamma, omega):
        # Unpack state
        S, E, I, R = y[0], y[1], y[2], y[3]
        # Rates
        dS = -beta * S * I + omega * R
        dE = beta * S * I - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I - omega * R
        # Return new state
        return (dS, dE, dI, dR)

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]
        beta, sigma, gamma, omega = (
            p["beta"],
            p["sigma"],
            p["gamma"],
            p["omega"],
        )

        # Wrapper to match solver API
        def f(t, y):
            return np.array(
                self._seirs(t, y, beta, sigma, gamma, omega),
                dtype=np.float64,
            )

        # Heal from scipy's dictming of kwargs
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        return solve_ivp(
            f,
            (t0, t1),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10,
            dense_output=debug,
        )