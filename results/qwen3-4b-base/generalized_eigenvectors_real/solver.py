from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        A, B = problem
        n = A.shape[0]
        L = np.linalg.cholesky(B)
        Linv = np.linalg.solve(L, np.eye(n))
        Atilde = Linv @ A @ Linv.T
        eigenvalues, eigenvectors = np.linalg.eigh(Atilde)
        eigenvectors = Linv.T @ eigenvectors
        norms_sq = np.diag(eigenvectors.T @ B @ eigenvectors)
        norms = np.sqrt(norms_sq)
        eigenvectors = eigenvectors / norms[:, None]
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = [eigenvectors[:, i].tolist() for i in range(eigenvectors.shape[1])]
        return (eigenvalues_list, eigenvectors_list)