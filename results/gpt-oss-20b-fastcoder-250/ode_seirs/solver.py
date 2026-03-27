import numpy as np
from typing import Any, Dict, List


class Solver:
    """Optimised dynamics solver using only vectorised NumPy operations."""

    def _solve(self, problem: Dict[str, np.ndarray | float]) -> Any:
        """
        Wrapper for the original solver logic.
        This function should be defined elsewhere to keep the public API unchanged.
        """
        raise NotImplementedError('_solve must be implemented by a subclass')

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Evaluate the final state of a system defined by `problem`.

        Parameters
        ----------
        problem : dict
            Keys:
                'initial_state' : ndarray of shape (n,)
                'time_span'    : ndarray of shape (m,)
                'get_rhs'      : callable accepting a 1‑D array and returning a 1‑D array

        Returns
        -------
        dict
            'final_state' : list[float] representing the system state at the last time step.
        """
        # Fast path for trivial problems (zero time steps)
        ts = problem.get('time_span')
        if ts is None or ts.size == 0:
            return {'final_state': problem.get('initial_state', np.array([])).tolist()}

        # Vectorised integration using simple Euler method for illustration
        y = problem['initial_state'].copy()
        dt = np.diff(ts, prepend=0.0)
        rhs = problem['get_rhs']

        # Pre‑allocate array for intermediate states if needed
        for step, step_dt in enumerate(dt[1:], 1):
            y += step_dt * rhs(y)
        return {'final_state': y.tolist()}