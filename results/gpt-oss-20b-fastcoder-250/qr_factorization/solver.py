import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Compute the QR factorization of the given matrix A (shape n x (n+1)).
        Returns {"QR": {"Q": Q.tolist(), "R": R.tolist()}}.
        """
        A = np.asarray(problem["matrix"])
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
