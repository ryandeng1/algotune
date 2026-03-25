import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]], **kwargs) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the real QZ factorization of the matrix pair (A, B).

        Parameters
        ----------
        problem : dict
            Dictionary containing the matrices "A" and "B" as nested lists.

        Returns
        -------
        dict
            Dictionary with key "QZ" containing the factors "AA", "BB", "Q", and "Z"
            as nested lists.
        """
        # Convert input lists to NumPy arrays
        A = np.array(problem["A"])
        B = np.array(problem["B"])

        # Perform the real QZ decomposition
        AA, BB, Q, Z = qz(A, B, output="real")

        # Convert the result arrays back to nested lists for serialization
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist()
            }
        }
