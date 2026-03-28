import numpy as np

class Solver:
    """
    Optimised solver for the principal matrix square root problem.
    Uses numpy's eigen decomposition which is significantly faster than
    scipy.linalg.sqrtm for dense square matrices.
    """
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        A = problem["matrix"]
        # Attempt eigen-decomposition
        try:
            w, v = np.linalg.eig(A)
            # np.linalg.eig may return complex eigenvalues/vectors; take principal sqrt
            sqrt_w = np.sqrt(w)
            # Reconstruct sqrtm: X = V * sqrt(D) * V^{-1}
            X = v @ np.diag(sqrt_w) @ np.linalg.inv(v)
            # Ensure numerical stability: round very small imaginary parts close to zero
            X = np.where(np.abs(X.real) < 1e-12, 0, X.real) + 1j * np.where(np.abs(X.imag) < 1e-12, 0, X.imag)
        except Exception:
            # Fallback to scipy in worst case
            try:
                from scipy.linalg import sqrtm
                X = sqrtm(A, disp=False)
            except Exception:
                X = np.empty((0, 0), dtype=complex)

        return {"sqrtm": {"X": X.tolist()}}