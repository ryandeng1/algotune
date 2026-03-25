import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray], **kwargs) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B.
        Uses scipy.linalg.eigh which directly handles the SPD metric matrix B.
        The result is sorted in descending eigenvalue order and eigenvectors are
        normalized with respect to the B-inner product.
        """
        A, B = problem

        # Compute eigenvalues and eigenvectors for the generalized symmetric
        # eigenvalue problem A x = λ B x.
        # eigh returns them in ascending order, columns are orthonormal with B.
        lam, vec = eigh(A, B, eigvals_only=False, subset_by_index=None)

        # Reverse order for descending eigenvalues
        lam = lam[::-1]
        vec = vec[:, ::-1]

        # Ensure that each eigenvector is B-normalized (it should already be, but
        # rounding errors may accumulate). Normalize explicitly.
        Bv = B @ vec
        norms = np.sqrt(np.sum(vec * Bv, axis=0))
        vec = vec / norms

        # Convert to python lists
        eigenvalues_list = lam.tolist()
        eigenvectors_list = [vec[:, i].tolist() for i in range(vec.shape[1])]

        return eigenvalues_list, eigenvectors_list
