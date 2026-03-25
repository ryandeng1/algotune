import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs):
        """
        Compute the Cholesky decomposition of a symmetric positive definite matrix A.
        Return a dictionary in the required format.

        Parameters
        ----------
        problem : dict
            Dictionary with a key "matrix" containing a 2D numpy array (or array‑like).

        Returns
        -------
        dict
            {"Cholesky": {"L": <list of lists of floats>}}
        """
        # Extract the matrix, ensuring it is a NumPy array
        A = np.asarray(problem["matrix"], dtype=np.float64)

        # Compute the Cholesky factorization
        L = np.linalg.cholesky(A)

        # Convert to list of lists for the specified output format
        L_list = L.tolist()

        return {"Cholesky": {"L": L_list}}
