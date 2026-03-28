import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorization of the matrix A.
        Uses SciPy for a faster backend when available, otherwise falls back to NumPy.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary with a single key 'matrix' containing the input array.

        Returns
        -------
        dict[str, dict[str, list[list[float]]]]
            Dictionary containing the Q and R matrices as nested Python lists.
        """
        A = problem["matrix"]

        # Prefer scipy.linalg.qr if it is available and offers a speed boost
        try:
            from scipy.linalg import qr  # local import to avoid costly global import
            Q, R = qr(A, mode="reduced")
        except Exception:
            Q, R = np.linalg.qr(A, mode="reduced")

        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}