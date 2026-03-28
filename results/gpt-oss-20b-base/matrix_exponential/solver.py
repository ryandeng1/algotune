from typing import Dict, List
import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Compute the matrix exponential of the given square matrix A.

        Parameters
        ----------
        problem : dict
            Dictionary containing one key 'matrix' with a 2‑D numpy array as its value.

        Returns
        -------
        dict
            Dictionary with key 'exponential' holding the matrix exponential
            represented as a plain list of lists of floats.
        """
        A = problem['matrix']
        # Compute exp(A) using the highly‑optimized SciPy routine.
        expA = expm(A)
        # Convert the NumPy array to a plain list of lists for the required output format.
        return {"exponential": expA.tolist()}