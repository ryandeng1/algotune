from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(
        self, problem: Tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for
        symmetric A and symmetric positive‑definite B using only
        NumPy to avoid the overhead of SciPy's ``eigh``.

        The returned eigenvalues are sorted in descending order and
        the eigenvectors are B‑orthonormal (vᵀ B v = 1).

        Parameters
        ----------
        problem:
            Tuple containing the symmetric matrix `A` and the symmetric
            positive‑definite matrix `B`.

        Returns
        -------
        eigenvalues:
            List of eigenvalues in descending order.
        eigenvectors:
            List of eigenvectors (each a list of floats) in the same order.
        """
        A, B = problem
        n = A.shape[0]

        # ------------------------------------------------------------------
        # 1. Reduce to a standard symmetric eigenvalue problem
        #    L = chol(B),   Atilde = L⁻¹ A L⁻ᵀ
        # ------------------------------------------------------------------
        L = np.linalg.cholesky(B)
        # solve L * X = Y  =>  X = L⁻¹ Y  using forward substitution
        # First       X1 = L⁻¹ A       (n×n)
        X1 = np.linalg.solve(L, A)
        # Then  Atilde = X1 * L⁻ᵀ  using backward substitution on X1ᵀ
        Atilde = np.linalg.solve(L.T, X1.T).T

        # ------------------------------------------------------------------
        # 2. Solve the standard problem Atilde * y = λ y
        # ------------------------------------------------------------------
        e_vals, y = np.linalg.eigh(Atilde)  # y: eigenvectors as columns

        # ------------------------------------------------------------------
        # 3. Transform back to obtain eigenvectors of the original problem
        #    v = L⁻ᵀ y   (solve for each column)
        # ------------------------------------------------------------------
        # Solve Lᵀ v = y  =>  v = (Lᵀ)⁻¹ y  using backward substitution
        v = np.linalg.solve(L.T, y)

        # ------------------------------------------------------------------
        # 4. Normalize so that vᵀ B v = 1 for each column
        #    Using the original B avoids computing L⁻¹ again
        # ------------------------------------------------------------------
        Bv = B @ v                # shape (n, n)
        norms = np.sqrt(np.einsum("ij,ij->j", v, Bv))  # shape (n,)
        v = v / norms

        # ------------------------------------------------------------------
        # 5. Order results in descending eigenvalue order
        # ------------------------------------------------------------------
        reverse = slice(None, None, -1)
        eigenvalues = e_vals[reverse].tolist()
        eigenvectors = [v[:, i].tolist() for i in range(n)][::-1]

        return eigenvalues, eigenvectors