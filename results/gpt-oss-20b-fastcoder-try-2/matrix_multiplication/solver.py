import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        """
        Multiply matrices A and B (both are lists of lists) and return
        the result as a list of lists.

        Parameters
        ----------
        problem: dict
            Dictionary with keys "A" and "B" containing the matrices
            to be multiplied.

        Returns
        -------
        list[list[float]]
            The product matrix A · B.
        """
        # Convert to NumPy arrays without copying if possible
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        # Use the high‑performance BLAS call via @ (np.matmul)
        C = A @ B

        # Return as a native Python list (for compatibility with the benchmark)
        return C.tolist()