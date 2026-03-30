import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# Pre-compile the RHS with Numba for speed
@njit
def _brusselator_rhs(t, y, A, B):
    X, Y = y[0], y[1]
    dX_dt = A + X * X * Y - (B + 1) * X
    dY_dt = B * X - X * X * Y
    return np.array([dX_dt, dY_dt])

class Solver:
    """
    Optimized solver for the Brusselator ODE system.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the ODE defined in `problem`.  
        The problem must contain:
            - 'y0': initial state, array-like of length 2
            - 't0': initial time (float)
            - 't1': final time (float)
            - 'params': dict with keys 'A' and 'B' (floats)

        Returns a dictionary mapping the variable names to a list of the final state values.
        """
        sol = self._solve(problem, debug=False)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Convert final state to standard Python list with names
        return {
            "X": [float(sol.y[0, -1])],
            "Y": [float(sol.y[1, -1])]
        }

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        """
        Core solver that uses a JIT‑compiled RHS for performance.
        If `debug` is True the solver evaluates the solution on a dense time grid;
        otherwise it only returns the dense output at the final time step.
        """
        # Convert inputs to np arrays of correct type
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        A, B = problem["params"]["A"], problem["params"]["B"]

        def rhs(t, y):
            return _brusselator_rhs(t, y, A, B)

        # Solver settings
        rtol = 1e-9
        atol = 1e-9
        method = "DOP853"  # Higher order, fewer steps for stiff behaviour

        # Provide dense evaluation only if requested
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol