from typing import Any, Dict
import numpy as np

class Solver:

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorisation of the input matrix A.
        Results are returned as Python lists for compatibility.
        """
        A = problem["matrix"]
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}