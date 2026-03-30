import numpy as np

class Solver:
    """Fast orthogonal Procrustes solver using NumPy's SVD."""

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Compute the orthogonal matrix G that best aligns A to B in the sense of
        minimizing ‖G A − B‖₂, where G is orthogonal.

        Parameters
        ----------
        problem : dict
            Must contain 'A' and 'B', both numpy arrays of identical shape.

        Returns
        -------
        dict
            ``{'solution': G}`` where G is an orthogonal matrix (numpy.ndarray).
        """
        # Fast array conversion (no copy if already ndarray)
        A = np.asarray(problem.get('A'))
        B = np.asarray(problem.get('B'))

        if A.shape != B.shape:
            return {}

        # Compute B Aᵀ
        M = B @ A.T

        # Singular value decomposition (full_matrices=False gives a
        # minimal‑size |U| and |V| for rectangular matrices)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Optimal orthogonal matrix
        G = U @ Vt

        return {"solution": G}