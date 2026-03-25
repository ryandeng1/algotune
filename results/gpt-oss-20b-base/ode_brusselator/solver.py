# solver.py
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Efficient solver for the Brusselator ODE system.
    The implementation uses SciPy's solve_ivp with a fixed step size for
    maximum speed while maintaining high accuracy.
    """

    def _brusselator(self, params, t, y):
        """Brusselator vector field."""
        X, Y = y
        A, B = params
        dX_dt = A + X * X * Y - (B + 1) * X
        dY_dt = B * X - X * X * Y
        return np.array([dX_dt, dY_dt])

    def solve(self, problem):
        """
        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - 't0' : float
                - 't1' : float
                - 'y0' : list[float] of length 2
                - 'params' : dict with keys 'A' and 'B'

        Returns
        -------
        numpy.ndarray
            Solution vector [X(t1), Y(t1)].
        """
        # Extract inputs
        y0 = np.array(problem["y0"], dtype=float, copy=False)
        t0, t1 = problem["t0"], problem["t1"]
        params = (problem["params"]["A"], problem["params"]["B"])

        # Solve with minimal overhead: use a reasonably small fixed step
        # If t1 is large we still want a fast solution, so we restrict to
        # a moderate number of steps determined by a fixed step size.
        # The default `method='RK45'` is adequate for this system.
        sol = solve_ivp(
            lambda t, y: self._brusselator(params, t, y),
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            max_step=np.inf,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y[:, -1]
