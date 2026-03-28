import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the Cholesky factorization of a symmetric positive‑definite matrix.

        Parameters
        ----------
        problem : dict
            Dictionary with key `'matrix'` containing a 2‑D NumPy array.

        Returns
        -------
        dict
            Nested dictionary ``{'Cholesky': {'L': L.tolist()}}`` where
            ``L`` is the lower‑triangular Cholesky factor.
        """
        # Direct call to the highly optimised BLAS/LAPACK routine via numpy
        L = np.linalg.cholesky(problem['matrix'])
        return {'Cholesky': {'L': L.tolist()}}