from typing import Any, Tuple, List
import numpy as np
from numpy.linalg import solve_triangular
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray]) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the generalized eigenvalue problem A · x = λ B · x for symmetric A and
        symmetric positive‑definite B using only NumPy. The routine returns the
        eigenvalues in descending order together with the corresponding eigenvectors.
        """
        A, B = problem
        n = A.shape[0]

        # 1. Cholesky factorization of B  :  B = L Lᵀ
        L = np.linalg.cholesky(B)          # shape (n, n)
        # 2. Transform to a standard problem
        #    Atilde = L⁻¹ A L⁻ᵀ   solved without forming the inverse
        X = solve_triangular(L.T, A, lower=False, check_finite=False)   # solve Lᵀ X = A
        Atilde = solve_triangular(L, X, lower=True, check_finite=False)  # solve L Y = X

        # 3. Eigen decomposition of the symmetric matrix
        eigvals, eigvecs = np.linalg.eigh(Atilde)   # eigvecs columns

        # 4. Back‑transform eigenvectors to the original basis
        #    v = L⁻¹ w  ⇒ solve L v = w  (lower‑triangular)
        eigvecs = solve_triangular(L, eigvecs, lower=True, check_finite=False)

        # 5. Normalise with respect to the B‑inner product
        #    ||v||_B² = vᵀ B v = ||Lᵀ v||²
        #    Compute u = Lᵀ v by solving Lᵀ u = v
        u = solve_triangular(L.T, eigvecs, lower=False, check_finite=False)
        norms = np.linalg.norm(u, axis=0)
        # avoid division by zero
        inv_norms = np.where(norms > 0, 1.0 / norms, 0.0)
        eigvecs *= inv_norms

        # 6. Reverse ordering to obtain descending eigenvalues
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # 7. Convert to Python lists as required by the interface
        eigenvalues_list = eigvals.tolist()
        eigenvectors_list = [eigvecs[:, i].tolist() for i in range(n)]

        return eigenvalues_list, eigenvectors_list