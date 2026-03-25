import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Any, Tuple, List


class Solver:
    def solve(
        self, problem: Tuple[NDArray[np.float64], NDArray[np.float64]], **kwargs
    ) -> Tuple[List[complex], List[List[complex]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for real matrices A and B.
        Returns the eigenvalues (complex) sorted in descending order by real part then
        imaginary part, and the corresponding unit‐norm eigenvectors.
        """
        A, B = problem
        # Compute generalized eigenvalues and eigenvectors
        eigvals, eigvecs = la.eig(A, B)

        # Normalize eigenvectors to unit Euclidean norm
        norm = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        nonzero = norm > 1e-15
        eigvecs[:, nonzero] = eigvecs[:, nonzero] / norm[0, nonzero]

        # Prepare pairs and sort
        pairs = list(zip(eigvals, [eigvecs[:, i] for i in range(eigvals.size)]))
        pairs.sort(key=lambda p: (-p[0].real, -p[0].imag))

        sorted_vals, sorted_vecs = zip(*pairs)

        # Convert to Python lists
        eigenvalues_list = list(sorted_vals)
        eigenvectors_list = [list(vec) for vec in sorted_vecs]

        return (eigenvalues_list, eigenvectors_list)
