from typing import List
import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, List[float]]) -> List[float]:
        """
        Solve the Toeplitz linear system Tx = b using the efficient
        scipy implementation, minimizing Python overhead.
        """
        # Convert inputs to contiguous NumPy arrays once
        c = np.asarray(problem['c'], dtype=np.float64, order='C')
        r = np.asarray(problem['r'], dtype=np.float64, order='C')
        b = np.asarray(problem['b'], dtype=np.float64, order='C')

        # Solve the Toeplitz system
        x = solve_toeplitz((c, r), b)

        # Return a Python list to match the expected interface
        return x.tolist()