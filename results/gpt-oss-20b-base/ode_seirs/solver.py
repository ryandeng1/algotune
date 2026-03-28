import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """Optimised SEIRS ODE solver using a tightly‑coupled callback."""

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict, debug: bool = True):
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Cache parameter values to avoid dictionary lookup inside the solver
        beta, sigma, gamma, omega = (
            params["beta"],
            params["sigma"],
            params["gamma"],
            params["omega"],
        )

        # Explicit ODE function with minimal allocations
        def seirs(t, y):
            S, E, I, R = y
            SIR = beta * S * I
            dSdt = -SIR + omega * R
            dEdt = SIR - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            return np.array([dSdt, dEdt, dIdt, dRdt], dtype=np.float64)

        # Solver settings tuned for speed/accuracy balance
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10,
            dense_output=debug,
        )