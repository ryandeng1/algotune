import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the OPP instance by computing M = B @ A.T, its SVD, and returning G = U @ V^T.
        """
        A = problem.get('A')
        B = problem.get('B')
        if A is None or B is None or A.shape != B.shape:
            return {}

        # Use the most efficient dot product
        M = np.dot(B, A.T)
        # Compute U and Vt only, discard singular values
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        G = np.dot(U, Vt)
        return {'solution': G.tolist()}