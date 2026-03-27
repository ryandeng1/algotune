from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        """
        Compute C = A · B using NumPy for speed and return the result as a plain list of lists.
        """
        # Convert the input lists to NumPy arrays with float64 dtype for speed.
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        # Perform the matrix multiplication with NumPy's optimized routine.
        C = np.dot(A, B)

        # Convert the result back to a Python list of lists.
        return C.tolist()