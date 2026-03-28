import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorization of matrix A and return it as nested
        Python lists.  Internally NumPy stores the data in C‑contiguous blocks,
        so the conversion to Python lists is inexpensive.
        """
        A = problem["matrix"]
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}