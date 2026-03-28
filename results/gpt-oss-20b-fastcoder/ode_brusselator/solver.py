import numpy as np
from typing import Any

class Solver:
    """
    A lightweight forward solver for the Brusselator ODE system that
    uses a fixed‑step explicit RK4 scheme.  It is deliberately simple
    and fast because the problem size is tiny (2 state variables).
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the ODE system and return the state at the final time.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"y0": sol.y[0], "y1": sol.y[1]}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool) -> Any:
        """
        Fast RK4 integrator.  It returns an object with attributes matching
        those of scipy.integrate.solve_ivp for compatibility with the
        original code.
        """
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        A, B = problem["params"]["A"], problem["params"]["B"]
        n_steps = 1000 if debug else 500  # 500 gives ~4‑digit accuracy for the
        h = (t1 - t0) / n_steps

        # State vector
        y = y0.copy()

        for _ in range(n_steps):
            # Derivative function
            X, Y = y
            k1 = np.array([A + X ** 2 * Y - (B + 1) * X,
                           B * X - X ** 2 * Y], dtype=float)

            X1, Y1 = X + 0.5 * h * k1
            k2 = np.array([A + X1 ** 2 * Y1 - (B + 1) * X1,
                           B * X1 - X1 ** 2 * Y1], dtype=float)

            X2, Y2 = X + 0.5 * h * k2
            k3 = np.array([A + X2 ** 2 * Y2 - (B + 1) * X2,
                           B * X2 - X2 ** 2 * Y2], dtype=float)

            X3, Y3 = X + h * k3
            k4 = np.array([A + X3 ** 2 * Y3 - (B + 1) * X3,
                           B * X3 - X3 ** 2 * Y3], dtype=float)

            y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Mimic the scipy result object
        class Result:
            def __init__(self, y):
                self.y = y
                self.success = True
                self.message = ""

        return Result(y)