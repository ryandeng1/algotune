import numpy as np
from typing import Dict, Any

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Solve the Orthogonal Procrustes Problem (OPP) by computing the
        singular value decomposition of M = B @ A.T and returning G = U @ V.T
        where [U,_,V] = svd(M).

        Parameters
        ----------
        problem : dict
            Must contain numpy‑compatible arrays 'A' and 'B'.

        Returns
        -------
        dict
            {'solution': G.tolist()}  (G is a real orthogonal matrix)
        """
        A = np.asarray(problem.get('A'), dtype=float)
        B = np.asarray(problem.get('B'), dtype=float)

        # Quick validation
        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
            return {}

        # Compute the cross‑covariance matrix
        M = B @ A.T

        # Economy‑size SVD (faster when n > 1_000)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Optimal orthogonal transformation
        G = U @ Vt

        return {'solution': G.tolist()}