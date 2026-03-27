from typing import Any
import numpy as np
from scipy.linalg import qz


class Solver:
    def solve(
        self, problem: dict[str, list[list[float]]]
    ) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Compute the QZ factorization of matrices A and B using scipy's efficient
        implementation.  The input lists are converted to 2‑D NumPy arrays
        before calling ``scipy.linalg.qz`` with ``output='real'``.
        """
        # Convert input matrices to NumPy arrays with float dtype
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        # Perform QZ factorization (upper Hessenberg/tridiagonal form)
        AA, BB, Q, Z = qz(A, B, output="real")

        # Convert results back to plain Python lists for JSON serialisation
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }