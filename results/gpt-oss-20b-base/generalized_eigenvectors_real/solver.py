from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B
        using the transformation A~ = L⁻¹ A L⁻ᵀ, where B = L Lᵀ.
        Returns eigenvalues sorted in descending order and corresponding B‑orthonormal
        eigenvectors.
        """
        A, B = problem

        # Cholesky factorisation and solve instead of matrix inverse
        L = np.linalg.cholesky(B)
        # solve L * X = A for X, then multiply by Lᵀ
        X = np.linalg.solve(L, A)
        Atilde = X @ L.T

        eig_vals, eig_vecs = np.linalg.eigh(Atilde)

        # Back‑transform eigenvectors: x = L⁻ᵀ y
        eig_vecs = np.linalg.solve(L.T, eig_vecs)

        # Normalise w.r.t. B: norm = sqrt(vᵀ B v)
        norms = np.sqrt(np.einsum('ij,ij->i', eig_vecs, B @ eig_vecs))
        # Avoid division by zero
        nonzero = norms > 0
        eig_vecs[:, nonzero] /= norms[nonzero]

        # Reverse order to get descending eigenvalues
        eig_vals = eig_vals[::-1]
        eig_vecs = eig_vecs[:, ::-1]

        # Convert to lists
        eig_vals_list = eig_vals.tolist()
        eig_vecs_list = [eig_vecs[:, i].tolist() for i in range(eig_vecs.shape[1])]

        return eig_vals_list, eig_vecs_list