import numpy as np

class Solver:

    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero,
        measured by absolute value.
        """
        # Convert to a NumPy array in one pass
        mat = np.array(problem["matrix"], dtype=float)

        # Since the matrix is symmetric we can use eigvalsh which is faster and
        # returns real values.  It returns a sorted array by eigenvalue value,
        # not by absolute value, so we need a quick way to get the two
        # smallest |λ|.
        vals = np.linalg.eigvalsh(mat)

        # Use partition to locate the two smallest absolute values without a full sort
        # This is O(n) instead of O(n log n).
        idx = np.argpartition(np.abs(vals), 2)[:2]
        # Retrieve the corresponding eigenvalues
        closest = vals[idx]
        # Sort them by absolute value before returning
        return sorted(closest.tolist(), key=abs)