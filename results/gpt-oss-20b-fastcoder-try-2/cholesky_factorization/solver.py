import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """Directly apply NumPy's highly‑optimized Cholesky routine."""
        A = problem['matrix']
        # Ensure float64 for numerical stability in NumPy's BLAS/LAPACK
        if A.dtype != np.float64:
            A = A.astype(np.float64, copy=False)
        L = np.linalg.cholesky(A)
        return {"Cholesky": {"L": L}}