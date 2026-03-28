import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Compute the matrix exponential of the input matrix using SciPy's efficient
        implementation and return it as a list of lists.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing the key 'matrix' mapping to the input square
            matrix A.

        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary with key 'exponential' containing exp(A) as a list of
            lists.
        """
        A = problem['matrix']
        expA = expm(A)                         # Fast C-implemented routine
        return {'exponential': expA.tolist()}