import numpy as np
from scipy.linalg import expm as scipy_expm

class Solver:

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Compute the matrix exponential of 'matrix' using the most efficient
        available method for the size of the matrix.
        """
        A = problem['matrix']
        n, m = A.shape
        assert n == m, "Matrix must be square"

        # For very small matrices use scipy.expm (optimized)
        if n <= 20:
            expA = scipy_expm(A)
        else:
            # For larger matrices, eigen-decomposition is usually faster
            try:
                w, v = np.linalg.eig(A)
                # Avoid numeric problems with nearly singular v
                if np.linalg.norm(v - v[:, np.newaxis], axis=0).min() < 1e-12:
                    raise Exception("Ill‑conditioning")
                exp_w = np.exp(w)
                expA = v @ np.diag(exp_w) @ np.linalg.inv(v)
            except Exception:
                # Fall back to scipy
                expA = scipy_expm(A)

        return {"exponential": expA.tolist()}