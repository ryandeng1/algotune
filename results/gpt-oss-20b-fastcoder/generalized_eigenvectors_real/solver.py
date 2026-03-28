import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """Fast solver for A x = λ B x with A symmetric, B SPD."""
        A, B = problem

        # Cholesky factorisation of B: B = L Lᵀ
        L = np.linalg.cholesky(B)

        # Solve L y = x for y instead of forming L⁻¹
        # Compute Ã = L⁻¹ A L⁻T
        y = np.linalg.solve(L, A)
        Atilde = np.linalg.solve(L.T, y)

        # Symmetric eigen–decomposition of Ã
        eigenvalues, eigvecs = np.linalg.eigh(Atilde)

        # Transform back to eigenvectors of the original problem
        eigvecs = np.linalg.solve(L.T, eigvecs)

        # Normalise eigenvectors to be B‑orthonormal:
        #  vᵀ B v = ||Lᵀ v||²
        Lt = L.T
        norms = np.linalg.norm(Lt @ eigvecs, axis=0)
        eigvecs /= norms

        # Return in descending eigenvalue order
        inds = np.argsort(eigenvalues)[::-1]
        ev = eigenvalues[inds].tolist()
        evs = eigvecs[:, inds].T.tolist()  # each eigenvector as a list

        return ev, evs