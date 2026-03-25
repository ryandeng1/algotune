import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the orthogonal Procrustes problem.

        Computes G = U @ Vt where M = B @ A.T and M has SVD M = U @ S @ Vt.

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys "A" and "B" with n-by-n matrices.

        Returns
        -------
        dict
            Dictionary with a single key "solution" containing the optimal orthogonal matrix G
            as a nested list of floats.
        """
        A = problem.get("A")
        B = problem.get("B")

        if A is None or B is None:
            return {}

        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        if A.shape != B.shape:
            return {}

        # Compute M = B @ A.T
        M = B @ A.T

        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Optimal orthogonal matrix
        G = U @ Vt

        return {"solution": G.tolist()}
