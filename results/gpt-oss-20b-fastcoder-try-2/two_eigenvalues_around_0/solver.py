from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Find the two eigenvalues of the given symmetric matrix that are closest to zero.
        """
        # Convert to NumPy array only once
        mat = np.asarray(problem['matrix'], dtype=float)

        # Compute all eigenvalues in increasing order (fast for symmetric matrices)
        vals = np.linalg.eigvalsh(mat)

        # Find the indices of the two smallest absolute eigenvalues without full sort
        abs_vals = np.abs(vals)
        idx = np.argpartition(abs_vals, 2)[:2]

        # Extract, sort them by absolute value, and return as list
        closest = vals[idx]
        return sorted(closest.tolist(), key=abs)