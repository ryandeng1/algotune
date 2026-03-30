from __future__ import annotations

import numpy as np
from typing import List, Dict

class Solver:
    """
    Optimised solver to find the two eigenvalues of a symmetric matrix that are
    closest to zero.

    The implementation relies completely on NumPy's native C routines and
    avoids Python‑level sorting or loops wherever possible. The solver works
    in three steps:

    1. Compute all eigenvalues of the symmetric matrix using ``eigvalsh``.
    2. Find the indices of the two eigenvalues with the smallest absolute
       values using ``np.argpartition`` which does a linear‑time selection.
    3. Extract those two eigenvalues and sort them by absolute value.

    This eliminates redundant work compared with a straight ``np.sort`` call
    on the entire eigenvalue array and keeps the operation in compiled code.
    """

    def solve(self, problem: Dict[str, List[List[float]]]) -> List[float]:
        # Convert the input to a NumPy array of type float64
        matrix = np.asarray(problem["matrix"], dtype=np.float64)

        # Compute all eigenvalues efficiently; the matrix is symmetric
        eigenvalues = np.linalg.eigvalsh(matrix)

        # Find the indices of the two eigenvalues with the smallest absolute value
        # np.argpartition gives linear‑time selection on the |eigenvalues| array
        abs_vals = np.abs(eigenvalues)
        if eigenvalues.size > 2:
            idx = np.argpartition(abs_vals, 2)[:2]
        else:  # matrix of size 1 or 2
            idx = np.arange(eigenvalues.size)

        # Retrieve the corresponding eigenvalues
        selected = eigenvalues[idx]

        # Finally, sort them by absolute value before returning
        # np.argsort runs in C and is very fast
        order = np.argsort(np.abs(selected))
        return selected[order].tolist()