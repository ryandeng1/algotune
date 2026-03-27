from typing import Any, List
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Efficiently solve a linear system Ax = b using NumPy.
        """
        # Convert inputs to NumPy arrays with the minimal work:
        # np.asarray skips copying if the operand is already an array.
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # numpy.linalg.solve is a fast, direct solver that doesn't
        # materialize the inverse of A.
        x = np.linalg.solve(A, b)

        # Convert the solution back to a plain Python list for the caller.
        return x.tolist()