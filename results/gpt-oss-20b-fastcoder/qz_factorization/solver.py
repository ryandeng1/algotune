import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Compute the real QZ (generalized Schur) factorization of the matrix pair (A, B).
        The result is returned in the same dictionary format as the input,
        with all matrices converted to nested Python lists.

        Parameters
        ----------
        problem : dict
            A dictionary with keys 'A' and 'B', each mapping to a list of lists
            containing the matrix entries.

        Returns
        -------
        dict
            A dictionary of the form
            {"QZ": {"AA": [...], "BB": [...], "Q": [...], "Z": [...]}}
        """
        # Convert to NumPy arrays once.  Using dtype=float implicitly
        # keeps the data type consistent for qz and avoids unnecessary copies.
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)

        # Perform the real QZ factorization.
        AA, BB, Q, Z = qz(A, B, output="real")

        # Convert results back to nested lists for the required output format.
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }