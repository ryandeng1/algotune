import numpy as np
from scipy import sparse
from typing import List, Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[complex]:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        
        if N <= 100:
            A_dense = A.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            pairs = list(zip(eigenvalues, eigenvectors.T))
            pairs.sort(key=lambda x: -np.abs(x[0]))
            return [vec for _, vec in pairs[:k]]
        else:
            eigenvalues, eigenvectors = sparse.linalg.eigs(
                A,
                k=k,
                v0=np.ones(N, dtype=A.dtype),
                maxiter=N * 200,
                ncv=max(2 * k + 1, 20),
            )
            pairs = list(zip(eigenvalues, eigenvectors.T))
            pairs.sort(key=lambda x: -np.abs(x[0]))
            return [vec for _, vec in pairs[:k]]
