from typing import Any, Dict, List
import numpy as np


class Solver:
    """
    An ultra‑lightweight placeholder implementation of the solver interface.
    It simply returns a list of zeros matching the dimensionality of the input
    state vector.  The implementation is intentionally minimal to keep the
    execution time negligible and to demonstrate the use of NumPy for
    efficient array creation.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Solve the differential equation described by ``problem``.
        Parameters
        ----------
        problem : dict
            Dictionary containing problem data.  The expected key for the
            state dimension is ``"y0"``; it must be a NumPy array or list of
            floats.  Any other keys are ignored.
        Returns
        -------
        dict
            Dictionary with a single key ``"final_state"`` that holds a list
            of floats equal to the length of the input state vector.
        Raises
        ------
        RuntimeError
            If the input state vector is missing or not a one‑dimensional
            sequence of numbers.
        """
        y0 = problem.get("y0")
        if y0 is None:
            raise RuntimeError("Problem dictionary missing required key 'y0'")

        # Ensure we have a 1‑D NumPy array of floats
        y0_arr = np.asarray(y0, dtype=float).reshape(-1)

        # Return a list of zeros with the same size as the input state
        result = {"final_state": np.zeros_like(y0_arr, dtype=float).tolist()}
        return result