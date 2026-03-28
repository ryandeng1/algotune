import numpy as np
from scipy.linalg import eig
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        A, B = problem
        # scale both matrices by the square root of ||B||₂ for numerical stability
        scale = np.sqrt(np.linalg.norm(B, ord=2))
        A_scaled = A / scale
        B_scaled = B / scale

        # solve generalized eigenproblem
        eigvals, eigvecs = eig(A_scaled, B_scaled, left=False, right=True, check_finite=False)

        # normalize eigenvectors to unit norm
        norms = np.linalg.norm(eigvecs, axis=0)
        nonzero = norms > 1e-15
        eigvecs[:, nonzero] /= norms[nonzero]

        # sort by real part descending, then imag part descending
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # convert to lists
        eigval_list = eigvals.tolist()
        eigvec_list = [vec.tolist() for vec in eigvecs.T]
        return eigval_list, eigvec_list