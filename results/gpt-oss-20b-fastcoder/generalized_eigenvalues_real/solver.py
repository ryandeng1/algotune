from typing import Any, List, Tuple
import numpy as np
from numpy.typing import NDArray


class Solver:

    def solve(self, problem: Tuple[NDArray, NDArray]) -> List[float]:
        """
        Solve the generalized symmetric eigenvalue problem A · x = λ B · x.

        The algorithm uses
        * Cholesky decomposition of B: B = L · Lᵀ
        * Transformation to a standard eigenvalue problem
          Â = L⁻¹ · A · L⁻ᵀ with an efficient solve instead of matrix inversion
        * numpy.linalg.eigh to compute all eigenvalues of the symmetric Â.

        The returned eigenvalues are sorted in descending order.
        """
        A, B = problem

        # Cholesky factorisation of the positive‑definite matrix B
        L = np.linalg.cholesky(B)

        # Solve L x = A for x → M = L⁻¹ A, avoiding explicit inversion
        M = np.linalg.solve(L, A)

        # Symmetric equivalent: Â = L⁻¹ A L⁻ᵀ = M · Mᵀ (because M = L⁻¹ A)
        # Using transpose solve for the second term.
        Atilde = M @ np.linalg.solve(L.T, M.T)

        # eigh returns eigenvalues in ascending order; reverse to descending
        eigvals = np.linalg.eigh(Atilde)[0][::-1]
        return eigvals.tolist()