from typing import Any
import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Compute the principal matrix square root of a matrix using scipy.linalg.sqrtm.
        """
        # Retrieve the matrix from the problem dictionary
        A = problem["matrix"]

        try:
            # scipy.linalg.sqrtm returns a complex matrix if necessary
            X = scipy.linalg.sqrtm(A, disp=False)
        except Exception:
            # In case of failure return an empty solution
            return {"sqrtm": {"X": []}}

        # Convert the result to a nested list of complex numbers
        return {"sqrtm": {"X": X.tolist()}}