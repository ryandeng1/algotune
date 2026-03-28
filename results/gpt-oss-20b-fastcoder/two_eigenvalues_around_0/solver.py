from typing import List, Dict
import numpy as np

class Solver:

    def solve(self, problem: Dict[str, List[List[float]]]) -> List[float]:
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero.
        The function is optimized to avoid Python‑level sorting and to use
        NumPy's efficient C‑level routines.
        """
        # Convert to a NumPy array (memory‑efficient; dtype=float already)
        matrix = np.asarray(problem['matrix'], dtype=float)

        # Compute all eigenvalues of the symmetric matrix (fast on CPU)
        eig_vals = np.linalg.eigvalsh(matrix)

        # Find indices of the two eigenvalues with smallest absolute value
        # Using np.argpartition gives order‑independent access in O(n)
        abs_vals = np.abs(eig_vals)
        idx = np.argpartition(abs_vals, 2)[:2]

        # Extract them and sort by absolute value for deterministic output
        result = eig_vals[idx]
        result.sort(key=lambda x: abs(x))
        return result.tolist()