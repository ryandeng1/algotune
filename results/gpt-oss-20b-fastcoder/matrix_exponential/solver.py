import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Compute the matrix exponential of a real square matrix.
        Uses scipy.linalg.expm for robust and efficient computation
        and returns the result as a NumPy array.

        Parameters
        ----------
        problem : dict
            Dictionary containing a key 'matrix' mapped to a 2-D NumPy array.

        Returns
        -------
        dict
            Dictionary with a single key 'exponential' pointing to the
            computed matrix exponential as a NumPy array.
        """
        A = problem["matrix"]
        # scipy.linalg.expm is highly optimised for a wide range of sizes,
        # including sparse (via dense conversion) and normal matrices.
        expA = expm(A)
        return {"exponential": expA}