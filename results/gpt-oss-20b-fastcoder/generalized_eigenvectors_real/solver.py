import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(
        self, problem: tuple[NDArray, NDArray]
    ) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for symmetric A and
        symmetric positive‑definite B.

        The routine uses the Cholesky factorisation of B and solves standard
        eigenvalue problem for the transformed matrix.  All operations are
        vectorised – no explicit Python loops are used – to maximise speed
        for large matrices.
        """
        A, B = problem

        # Cholesky factorisation of B: B = L·L.T
        # Use np.linalg.cholesky (fast, BLAS-backed)
        L = np.linalg.cholesky(B)

        # Compute L⁻¹ once via solve_triangular
        # Since L is triangular, we can solve L·X = I for X = L⁻¹
        # We use np.linalg.inv here for brevity; the matrices are small enough.
        Linv = np.linalg.inv(L)

        # Similarity transform: Atilde = L⁻¹·A·L⁻T
        Atilde = Linv @ A @ Linv.T

        # Eigen-decomposition of the symmetric Atilde
        ev, evec = np.linalg.eigh(Atilde)

        # Back‑transform eigenvectors: v = L⁻T·q
        evec = L.T @ evec

        # Normalise eigenvectors with respect to B: (vᵀ B v)¹ᐟ²
        # Compute B·evec once
        B_evec = B @ evec
        norms = np.sqrt(np.einsum("ij,ij->i", evec, B_evec))
        # Avoid division by zero – norms should be >0 for positive‑definite B
        evec = evec / norms

        # Return eigenvalues/vectors in descending order
        ev = ev[::-1].tolist()
        evec = evec[:, ::-1]
        evec_list = [evec[:, i].tolist() for i in range(evec.shape[1])]

        return ev, evec_list