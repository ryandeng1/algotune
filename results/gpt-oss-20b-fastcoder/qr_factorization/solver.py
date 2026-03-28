import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorization of the input matrix.
        The results are returned as nested lists for compatibility with the
        expected output format.
        """
        A = problem["matrix"]
        # Short local references for speed
        qr = np.linalg.qr
        Q, R = qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}