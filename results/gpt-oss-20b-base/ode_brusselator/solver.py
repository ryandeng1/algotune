# solver.py
from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

# Numba will be compiled once at import time – it won't be counted in the runtime.
from numba import njit

# A fast, compiled version of the Brusselator RHS.
@njit
def _brusselator(t: float, y: np.ndarray, A: float, B: float) -> np.ndarray:
    X, Y = y[0], y[1]
    dX_dt = A + X * X * Y - (B + 1) * X
    dY_dt = B * X - X * X * Y
    return np.array([dX_dt, dY_dt], dtype=y.dtype)

class Solver:
    """
    Solver for the Brusselator ODE system.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the ODE on the time interval [t0, t1] with initial conditions y0
        and parameters A, B. Return the final state as a dict of lists.
        """
        # Extract data – no extra copies are made.
        y0 = problem["y0"]
        t0, t1 = problem["t0"], problem["t1"]
        A, B = problem["params"]["A"], problem["params"]["B"]

        # Wrapper to match the signature expected by solve_ivp.
        def rhs(t, y):
            return _brusselator(t, y, A, B)

        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Convert the final state to a list of floats for consistency with the
        # original interface which returns a dict of lists of floats.
        final_state = sol.y[:, -1].tolist()
        return {"final_state": [float(v) for v in final_state]}

    # Expose the raw solver for debugging or testing purposes.
    def _solve(self, problem: Dict[str, Any], debug: bool = False) -> Any:
        y0 = np.array(problem['y0'])
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']

        def brusselator(t, y):  # pragma: no cover
            A, B = params['A'], params['B']
            X, Y = y
            dX_dt = A + X**2 * Y - (B + 1) * X
            dY_dt = B * X - X**2 * Y
            return np.array([dX_dt, dY_dt])

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            brusselator,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=t_eval,
            dense_output=debug,
        )