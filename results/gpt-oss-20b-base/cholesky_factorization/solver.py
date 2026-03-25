# solver.py
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the Cholesky factorization of a symmetric positive definite matrix.

        Parameters
        ----------
        problem : dict
            Dictionary containing the key "matrix" with a numpy.ndarray or list of lists
            representing a symmetric positive definite matrix A.

        Returns
        -------
        dict
            Dictionary with key "Cholesky" mapping to a dictionary containing key "L":
            a lower triangular matrix as a list of lists.
        """
        # Extract matrix and ensure it is a NumPy array
        A = np.array(problem["matrix"], dtype=np.float64)

        # Compute the lower triangular Cholesky factor
        L = np.linalg.cholesky(A)

        # Convert to a nested list for output
        L_list = L.tolist()

        return {"Cholesky": {"L": L_list}}
