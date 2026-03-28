import numpy as np
from scipy.linalg import lu_factor

class Solver:
    """
    Optimised LU-factorisation solver.

    This implementation uses :func:`scipy.linalg.lu_factor` which
    performs the factorisation in a single pass and avoids the overhead
    of creating the permutation matrix explicitly.  Post‑processing
    reconstructs the required ``P``, ``L`` and ``U`` matrices only once
    before converting them to Python lists for the final output.
    """

    @staticmethod
    def _build_permutation(piv: np.ndarray) -> np.ndarray:
        """
        Convert pivot array from ``scipy.linalg.lu_factor`` to a permutation matrix.

        Parameters
        ----------
        piv : np.ndarray
            1‑based pivot indices returned by ``lu_factor``.
        """
        n = piv.shape[0]
        # pivot array is 1‑based and contains row swaps
        perm = np.arange(n)
        for i in range(n):
            j = piv[i] - 1
            if i != j:
                perm[i], perm[j] = perm[j], perm[i]
        P = np.eye(n)[perm]
        return P

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem['matrix']
        lu, piv = lu_factor(A)

        # Construct permutation matrix
        P = self._build_permutation(piv)

        # Extract L and U from the combined matrix
        n = lu.shape[0]
        L = np.tril(lu, -1) + np.eye(n)
        U = np.triu(lu)

        # Convert to lists for the expected output format
        return {
            'LU': {
                'P': P.tolist(),
                'L': L.tolist(),
                'U': U.tolist()
            }
        }