from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        A, B = problem
        L = np.linalg.cholesky(B)
        Linv = np.linalg.inv(L)
        Atilde = Linv @ A @ Linv.T
        eigenvalues, eigenvectors = np.linalg.eigh(Atilde)
        eigenvectors = Linv.T @ eigenvectors
        B_eig = B @ eigenvectors
        norm_sq = eigenvectors.T @ B_eig
        norms = np.sqrt(norm_sq.diagonal())
        eigenvectors = eigenvectors / norms[:, np.newaxis]
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        return (eigenvalues.tolist(), [eigenvectors[:, i].tolist() for i in range(eigenvectors.shape[1])])