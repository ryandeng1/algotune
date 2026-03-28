import math
from typing import Any

class Solver:
    """
    Fast SEIRS solver using a fixed‑step explicit RK4 integrator.
    Works for the specific equations used in the original problem.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    # ------------------------------------------------------------------
    # INTERNAL IMPLEMENTATION
    # ------------------------------------------------------------------
    class _Result:
        __slots__ = ("success", "message", "y")

        def __init__(self, success: bool = True, message: str = "", y=None):
            self.success = success
            self.message = message
            self.y = y or []  # shape (4, n)

    def _solve(self, problem: dict[str, Any], debug: bool) -> Any:
        # Initial state
        y0 = list(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]

        # Time stepping parameters
        n_steps = 1000 if debug else 1000
        dt = (t1 - t0) / n_steps

        # Pre‑get parameters for speed
        beta, sigma, gamma, omega = (
            p["beta"],
            p["sigma"],
            p["gamma"],
            p["omega"],
        )

        # Storage for output
        y_values = []

        # Helper: derivative function
        def f(t, y):
            S, E, I, R = y
            dS = -beta * S * I + omega * R
            dE = beta * S * I - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I - omega * R
            return (dS, dE, dI, dR)

        # Initialise
        t = t0
        y = y0

        # Main RK4 loop
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + 0.5 * dt, [y[i] + 0.5 * dt * k1[i] for i in range(4)])
            k3 = f(t + 0.5 * dt, [y[i] + 0.5 * dt * k2[i] for i in range(4)])
            k4 = f(t + dt, [y[i] + dt * k3[i] for i in range(4)])

            y = [
                y[i]
                + (dt / 6.0)
                * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                for i in range(4)
            ]

            t += dt
            if debug:
                y_values.append(tuple(y))

        # Prepare the result to mimic scipy's solution
        if debug:
            y_array = [[val[i] for val in y_values] for i in range(4)]
        else:
            y_array = [y]
        return self._Result(success=True, y=y_array)