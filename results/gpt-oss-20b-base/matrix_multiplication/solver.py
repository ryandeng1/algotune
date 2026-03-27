from typing import List, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]]) -> List[List[float]]:
        """
        Compute C = A · B using NumPy's efficient BLA implementation.
        The input matrices are converted to contiguous float64 arrays for
        maximum performance, and the result is returned as a list of lists.
        """
        # Convert the nested Python lists to contiguous NumPy arrays
        A = np.asarray(problem["A"], dtype=np.float64, order="C")
        B = np.asarray(problem["B"], dtype=np.float64, order="C")

        # Use the highly optimised BLAS matmul routine
        C = A @ B

        # Convert back to Python lists (no copy of data)
        return C.tolist()