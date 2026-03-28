from typing import Any
import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Solve the QZ factorization problem by computing the QZ factorization of (A,B).
        Uses scipy.linalg.qz with mode='real' to compute:
            A = Q AA Z*
            B = Q BB Z*
        """
        # Convert input lists to numpy arrays without copying data when possible
        A = np.asarray(problem["A"], dtype=np.float64, order="C")
        B = np.asarray(problem["B"], dtype=np.float64, order="C")

        # Perform the QZ factorization in real mode (the default)
        AA, BB, Q, Z = qz(A, B, output="real")

        # Convert numpy arrays back to Python lists for the required output format
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }