import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorization of the matrix A.
        :param problem: Dictionary containing the key 'matrix' with a 2-D np.ndarray.
        :return: Dictionary {"QR": {"Q": Q list, "R": R list}}.
        """
        A = problem["matrix"]
        # Compute reduced QR once using a local reference for speed
        qr_func = np.linalg.qr
        Q, R = qr_func(A, mode="reduced")
        # Convert to plain Python lists in a single step
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}