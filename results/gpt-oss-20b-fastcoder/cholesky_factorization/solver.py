from typing import Dict, Any
import numpy as np

class Solver:
    # Use __slots__ to reduce per-instance memory overhead
    __slots__ = ()

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, list]]:
        """
        Compute the Cholesky decomposition of a symmetric positive‑definite matrix.

        Parameters
        ----------
        problem
            Dictionary containing the key 'matrix' mapped to a NumPy
            ndarray of shape (n, n).

        Returns
        -------
        dict
            Contains a single key 'Cholesky' whose value is another
            dictionary with key 'L' pointing to a list of lists that
            represent the lower‑triangular Cholesky factor.
        """
        A = problem["matrix"]
        # NumPy's LAPACK routine is highly optimised for this operation.
        L = np.linalg.cholesky(A).tolist()
        return {"Cholesky": {"L": L}}