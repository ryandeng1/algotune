import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the matrix exponential exp(A) for the given square matrix A.
        The input matrix is given under the key "matrix" in the problem dictionary.
        The output is a dictionary with key "exponential" containing the resulting matrix
        as a nested list of floats.

        Parameters
        ----------
        problem : dict
            Expected to contain a key "matrix" whose value is a 2-D array-like
            representation of the square matrix to exponentiate.

        Returns
        -------
        dict
            {"exponential": <list of lists>} where the inner lists represent the
            rows of exp(A).
        """
        # Convert the input to a NumPy array (fast conversion from list of lists)
        A = np.asarray(problem["matrix"], dtype=np.float64)

        # Compute the matrix exponential using SciPy's highly optimized routine
        expA = expm(A)

        # Convert the result back to plain Python nested lists to satisfy the API
        result = {"exponential": expA.tolist()}

        return result
