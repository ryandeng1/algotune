import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        A, B = problem
        scale = np.sqrt(np.linalg.norm(B, ord='fro'))  # use frobenius norm for better scaling
        A /= scale
        B /= scale

        eigvals, eigvecs = eig(A, B, left=False, select='a')

        # Normalise eigenvectors
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        eigvecs = eigvecs / np.where(norms > 1e-15, norms, 1)

        # Sort by real then imaginary part descending
        idx = np.lexsort((-eigvals.imag, -eigvals.real))  # lexsort takes keys from last to first
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return (list(eigvals), [list(col) for col in eigvecs.T])