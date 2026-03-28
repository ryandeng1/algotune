import numpy as np

class Solver:
    """Fast LU decomposition without external heavy libraries."""

    @staticmethod
    def _lu_factor(A: np.ndarray):
        """
        Computes the LU decomposition with partial pivoting using only NumPy.
        Returns P (permutation matrix), L, and U as NumPy arrays.
        """
        n = A.shape[0]
        U = A.copy().astype(float)
        L = np.eye(n, dtype=float)
        P = np.eye(n, dtype=float)

        for k in range(n):
            # Pivot selection
            pivot = np.argmax(np.abs(U[k:, k])) + k
            if pivot != k:
                # Swap rows in U
                U[[k, pivot], :] = U[[pivot, k], :]
                # Swap rows in P
                P[[k, pivot], :] = P[[pivot, k], :]
                if k > 0:
                    # Swap rows in L but only columns before k
                    L[[k, pivot], :k] = L[[pivot, k], :k]

            pivot_val = U[k, k]
            if abs(pivot_val) < 1e-14:  # Handle zero pivot
                continue
            # Compute multipliers
            L[k + 1 : n, k] = U[k + 1 : n, k] / pivot_val
            # Eliminate below
            U[k + 1 : n, k + 1 :] -= L[k + 1 : n, k][:, None] * U[k, k + 1 :]

        return P, L, U

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        P, L, U = self._lu_factor(A)
        return {
            "LU": {
                "P": P.tolist(),
                "L": L.tolist(),
                "U": U.tolist()
            }
        }