from typing import Any
import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Compute the QZ factorization of a pair of real matrices (A, B) using
        ``scipy.linalg.qz`` with ``output='real'``.
        """
        A = np.array(problem['A'], dtype=float, copy=False)
        B = np.array(problem['B'], dtype=float, copy=False)

        # Compute factorization
        AA, BB, Q, Z = qz(A, B, output='real')

        # Convert to plain Python lists for the expected return format
        return {
            'QZ': {
                'AA': AA.tolist(),
                'BB': BB.tolist(),
                'Q': Q.tolist(),
                'Z':   Z.tolist()
            }
        }