from typing import Any
import numpy as np
import scipy.linalg


class Solver:
    """
    Optimised solver for the principal matrix square root problem.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Compute the principal square root of the given matrix using scipy.linalg.sqrtm.

        Parameters
        ----------
        problem : dict
            Dictionary with single key "matrix" pointing to a numpy.ndarray (2‑D).

        Returns
        -------
        dict
            {"sqrtm": {"X": <list of lists containing complex numbers>}}
            If the calculation fails, the returned list will be empty.
        """
        A = problem["matrix"]

        # scipy.linalg.sqrtm internally uses complex doubles; no need to cast here.
        # disp=False suppresses any warnings that would otherwise be printed.
        try:
            X, _ = scipy.linalg.sqrtm(A, disp=False)
        except Exception:
            # Return a consistent failure representation
            return {"sqrtm": {"X": []}}

        # Convert to plain Python lists so the result is JSON serialisable.
        return {"sqrtm": {"X": X.tolist()}}