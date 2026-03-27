import numpy as np
from typing import Dict, List, Any


class Solver:
    """
    A minimal but efficient ODE solver implementation based on the
    explicit Euler method. The solver accepts a problem dictionary
    describing the initial state `y0`, a callable for the derivative
    function `fun`, the time span `t_span`, and the number of steps
    `n_steps`. The result is returned in the required dictionary format.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the ODE specified in *problem* and return the final state.

        Parameters
        ----------
        problem : dict
            Must contain:
                * 'y0'   : np.ndarray   – Initial state vector.
                * 'fun'  : callable     – derivative function f(t, y).
                * 't_span' : tuple[float, float] – Start and end time.
                * 'n_steps' : int        – Number of integration steps.

        Returns
        -------
        dict
            Dictionary with a single key 'final_state' mapping to the list
            representation of the state at the final time step.
        """
        y0 = problem["y0"]
        f = problem["fun"]
        t0, t1 = problem["t_span"]
        n = problem["n_steps"]

        h = (t1 - t0) / n
        y = y0.astype(np.float64, copy=True)

        t = t0
        for _ in range(n):
            y += h * f(t, y)
            t += h

        return {"final_state": y.tolist()}