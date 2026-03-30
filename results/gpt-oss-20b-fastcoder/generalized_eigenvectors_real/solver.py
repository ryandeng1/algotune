from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Solver for the generalized eigenvalue problem A x = λ B x where
    A is symmetric and B is symmetric positive‑definite.

    The implementation follows the classical approach:
    1. Cholesky factorisation of B: B = L Lᵀ
    2. Solve the standard eigenvalue problem for the similarity
       transformed matrix Ĥ = L⁻¹ A L⁻ᵀ
    3. Recover the eigenvectors of the original problem: x = L⁻ᵀ v
    4. Normalise the eigenvectors with respect to B.
    5. Return the eigenvalues/vectors sorted in descending order.
    """

    def solve(
        self,
        problem: tuple[NDArray, NDArray]
    ) -> tuple[list[float], list[list[float]]]:
        """
        Solve A x = λ B x.

        Parameters
        ----------
        problem : tuple[NDArray, NDArray]
            (A, B) where A is symmetric and B is symmetric positive definite.

        Returns
        -------
        tuple[list[float], list[list[float]]]
            (eigenvalues, eigenvectors) with eigenvalues sorted in descending order.
        """
        A, B = problem

        # 1. Cholesky factorisation
        L = np.linalg.cholesky(B)          # B = L Lᵀ

        # 2. Solve for L⁻¹ without explicitly computing the inverse
        Linv = np.linalg.solve(L, np.eye(L.shape[0]))

        # 3. Similarity transformation
        A_tilde = Linv @ A @ Linv.T

        # 4. Standard eigensolver (sorted ascending)
        eigvals, eigvecs = np.linalg.eigh(A_tilde)

        # 5. Recover eigenvectors of the original problem
        eigvecs = Linv.T @ eigvecs

        # 6. Normalise with respect to B
        #    Using the Cholesky factor makes this cheap
        #    norm(v) = sqrt(vᵀ B v) = sqrt(vᵀ L Lᵀ v) = ||Lᵀ v||
        Linv_T = Linv.T
        norms = np.linalg.norm(Linv_T @ eigvecs, axis=0)
        # Avoid division by zero – norm should never be zero for valid B
        eigvecs /= norms

        # 7. Reverse to descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # 8. Convert to native Python lists for the required output format
        eigenvalues_list = eigvals.tolist()
        eigenvectors_list = [eigvecs[:, i].tolist() for i in range(eigvecs.shape[1])]

        return eigenvalues_list, eigenvectors_list