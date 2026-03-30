# solver.py

import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List

class Solver:
    """
    Fast solver for the Lotka–Volterra system using SciPy's solve_ivp.
    The implementation inlines parameters, avoids array allocations inside the
    ODE function and uses lightweight local variables for speed.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the ODE system and return the final state as a list.
        Raises RuntimeError if the solver fails.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"x": sol.y[0, -1], "y": sol.y[1, -1]}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, Any], debug: bool = True) -> Any:
        # Ensure inputs are numpy arrays (may already be)
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Pull parameters once to local variables for faster access
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])

        # Scalar ODE function – avoids numpy array creation each step
        def lotka_volterra(t: float, y):
            x, y_ = y  # rename to avoid shadowing
            dx = alpha * x - beta * x * y_
            dy = delta * x * y_ - gamma * y_
            return [dx, dy]  # list is faster than np.array in solve_ivp

        # Integration settings
        rtol = 1e-10
        atol = 1e-10
        method = "RK45"

        # If debugging we want a dense output and evaluation points
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        # Run the solver
        sol = solve_ivp(
            lotka_volterra,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol