import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the orthogonal Procrustes problem:
            min_G ||G A - B||_F^2   subject to G^T G = I.

        The optimal orthogonal matrix G is given by G = U V^T where
        M = B @ A^T = U S V^T is the singular value decomposition of M.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {"solution": []}

        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.shape != B.shape:
            return {"solution": []}

        # Compute M = B @ A.T
        M = B @ A.T

        # SVD of M
        U, _ , Vt = np.linalg.svd(M, full_matrices=False)

        # Orthogonal solution G
        G = U @ Vt

        return {"solution": G.tolist()}
