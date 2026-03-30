from typing import Dict, List
import numpy as np
from scipy.linalg import expm

class Solver:
    """
    Solver for computing the matrix exponential exp(A).
    Uses scipy's LAPACK-backed expm for speed.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Compute the matrix exponential of the given square matrix.

        Parameters
        ----------
        problem :
            Dictionary containing the key 'matrix' with a NumPy array.

        Returns
        -------
        dict
            A dictionary with key 'exponential' containing the matrix
            exponential as a list-of-lists of floats.
        """
        A = problem["matrix"]
        # Ensure the matrix is in the expected shape and type.
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square matrix.")
        # Compute the exponential using the highly-optimized LAPACK routine.
        expA = expm(A.astype(np.float64, copy=False))
        # Convert the result to the required list-of-lists format.
        return {"exponential": expA.tolist()}