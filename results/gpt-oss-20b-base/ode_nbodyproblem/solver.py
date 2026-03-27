from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np


class Solver:
    """
    A very small, self‑contained ODE solver that performs an explicit Euler
    integration for a system of linear ODEs of the form

        y' = A · y + b

    The input dictionary must contain the following keys:

        - ``A``:  2‑D NumPy array of shape (n, n).
        - ``b``:  1‑D NumPy array of length n.
        - ``y0``: 1‑D NumPy array of the initial state (length n).
        - ``t``:  1‑D NumPy array of time points at which the solution is
                 required.  The array must start with zero.

    The solver returns a dictionary containing:

        - ``success``: bool, always ``True`` for the simple Euler method.
        - ``y``:       2‑D NumPy array of shape (n, len(t)) with the state
                       trajectory.
        - ``message``: empty string.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Compute the state of the system at the final time point.
        Parameters
        ----------
        problem : dict
            Dictionary containing the keys described in the class docstring.

        Returns
        -------
        dict[str, list[float]]
            The state at the final time point represented as a list of floats.
        """
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64).reshape(-1)
        y0 = np.asarray(problem["y0"], dtype=np.float64).reshape(-1)
        t = np.asarray(problem["t"], dtype=np.float64).reshape(-1)

        if t[0] != 0.0:
            raise ValueError("Time array must start from 0.0")

        n = A.shape[0]
        steps = len(t)

        # Pre‑allocate trajectory matrix
        y = np.empty((n, steps), dtype=np.float64)
        y[:, 0] = y0

        # Compute time step sizes
        dt = np.diff(t)

        # Vectorised Euler integration
        for k in range(steps - 1):
            y_dot = A @ y[:, k] + b
            y[:, k + 1] = y[:, k] + dt[k] * y_dot

        result = {"success": True, "y": y, "message": ""}

        # Return final state as list of floats
        final_state = result["y"][:, -1].tolist()
        return {"success": result["success"], "final_state": final_state, "message": result["message"]}


# Example usage (to be removed or commented out in production):
if __name__ == "__main__":
    solver = Solver()
    prob = {
        "A": np.array([[-1.0, 0.0], [0.0, -0.5]]),
        "b": np.array([0.0, 0.0]),
        "y0": np.array([1.0, 2.0]),
        "t": np.linspace(0, 10, 101),
    }
    sol_dict = solver.solve(prob)
    print(sol_dict["final_state"])