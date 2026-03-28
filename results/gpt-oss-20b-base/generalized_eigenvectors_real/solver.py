import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for symmetric A and symmetric
        positive definite B using only NumPy primitives without explicit matrix inverses.
        The returned eigenvalues are sorted in descending order and eigenvectors are
        normalized with respect to B (i.e., xᵀ·B·x = 1).
        """
        A, B = problem

        # Cholesky decomposition B = L·Lᵀ
        L = np.linalg.cholesky(B)

        # Compute Ã = L⁻¹ · A · L⁻ᵀ without forming L⁻¹ explicitly
        # First solve L·Y = A  →  Y = L⁻¹·A
        Y = np.linalg.solve(L, A)
        # Then solve Lᵀ·Z = Yᵀ → Zᵀ = L⁻ᵀ·Y
        Z = np.linalg.solve(L.T, Y.T).T
        Atilde = Z

        # Standard eigenproblem on the transformed matrix
        eigvals, eigvecs = np.linalg.eigh(Atilde)

        # Back-transform eigenvectors: x = L⁻ᵀ·v
        eigvecs = np.linalg.solve(L.T, eigvecs)

        # Normalize eigenvectors with respect to B
        Bv = B @ eigvecs                    # B·x for all columns
        norms = np.sqrt(np.einsum('ij,ij->i', eigvecs, Bv))
        # Avoid division by zero – norm>0 by construction
        eigvecs /= norms

        # Return in descending order of eigenvalues
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        return (eigvals.tolist(), [eigvecs[:, i].tolist() for i in range(eigvecs.shape[1])])