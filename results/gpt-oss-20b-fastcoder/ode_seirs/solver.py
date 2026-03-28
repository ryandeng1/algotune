import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        beta, sigma, gamma, omega = (
            params["beta"],
            params["sigma"],
            params["gamma"],
            params["omega"],
        )

        def seirs(_: float, y: np.ndarray):
            S, E, I, R = y
            dSdt = -beta * S * I + omega * R
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            return np.array([dSdt, dEdt, dIdt, dRdt], dtype=float)

        # Keep tolerances at a reasonable level for speed
        rtol, atol = 1e-8, 1e-8

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )