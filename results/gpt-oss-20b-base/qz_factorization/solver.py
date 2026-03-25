# solver.py
import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Compute the real QZ factorization of the matrix pair (A, B).

        Parameters
        ----------
        problem : dict
            Dictionary containing two keys:
                - "A": n x n array_like
                - "B": n x n array_like
            The matrices must be square and of the same size.

        Returns
        -------
        dict
            Dictionary containing the key "QZ" which maps to another dictionary with
            keys "AA", "BB", "Q", and "Z".  Each value is a list of lists
            representing the corresponding matrix in row-major order.
        """
        # Convert input lists to numpy arrays
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)

        # Perform real QZ factorization
        AA, BB, Q, Z = qz(A, B, output="real")

        # Convert numpy arrays back to plain Python lists for JSON serialisation
        solution = {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist()
            }
        }
        return solution
