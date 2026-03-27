from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the differential equation system defined in `problem`.

        Parameters
        ----------
        problem : dict[str, np.ndarray | float]
            A dictionary containing the problem parameters.
            The keys expected by `_solve` are left unchanged.

        Returns
        -------
        dict[str, list[float]]
            Dictionary containing the final state of the system.

        Raises
        ------
        RuntimeError
            If the underlying solver fails.
        """
        # Delegate the heavy lifting to the private routine.
        sol = self._solve(problem, debug=False)

        # Extract the final state only if the solution succeeded.
        if sol.success:
            # `sol.y` is assumed to be a 2‑D NumPy array where the first axis
            # corresponds to state variables and the second to time points.
            # We return the last column as a plain Python list.
            return sol.y[:, -1].tolist()
        # If the solver failed, propagate a clear diagnostics message.
        raise RuntimeError(f"Solver failed: {sol.message}")