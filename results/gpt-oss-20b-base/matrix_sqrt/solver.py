import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Computes the principal matrix square root X of the matrix A in `problem["matrix"]`
        using an eigen-decomposition based approach.  If the decomposition fails,
        an empty result is returned to signal failure.
        """
        A = problem["matrix"]

        try:
            # Eigen-decomposition of the matrix
            eigvals, eigvecs = np.linalg.eig(A)
            # Construct the diagonal matrix of square roots of eigenvalues
            sqrt_diag = np.diag(np.sqrt(eigvals))
            # Compute the matrix square root
            X = eigvecs @ sqrt_diag @ np.linalg.inv(eigvecs)
        except Exception:
            return {"sqrtm": {"X": []}}

        return {"sqrtm": {"X": X.tolist()}}