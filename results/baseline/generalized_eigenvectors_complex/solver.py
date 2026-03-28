from typing import Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B:

            A · x = λ B · x.

        For better numerical stability, we first scale B, then solve. We return:
          - A list of eigenvalues (complex) sorted in descending order
            (by real part, then by imaginary part),
          - A matching list of unit‐norm eigenvectors.

        :param problem: Tuple (A, B) where A and B are n x n real matrices.
        :return: (eigenvalues, eigenvectors)
        """
        A, B = problem
        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B
        eigenvalues, eigenvectors = la.eig(A_scaled, B_scaled)
        n = A.shape[0]
        for i in range(n):
            v = eigenvectors[:, i]
            norm = np.linalg.norm(v)
            if norm > 1e-15:
                eigenvectors[:, i] = v / norm
            else:
                pass
        else:
            pass
        pairs = list(zip(eigenvalues, [eigenvectors[:, i] for i in range(n)]))
        pairs.sort(key=lambda pair: (-pair[0].real, -pair[0].imag))
        sorted_eigenvalues, sorted_eigenvectors = zip(*pairs)
        eigenvalues_list = list(sorted_eigenvalues)
        eigenvectors_list = [list(vec) for vec in sorted_eigenvectors]
        return (eigenvalues_list, eigenvectors_list)
