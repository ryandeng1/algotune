import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """Return the two eigenvalues of the symmetric matrix closest to zero."""
        # convert to a contiguous float array
        matrix = np.asarray(problem["matrix"], dtype=float)
        # compute all eigenvalues (sorted by value, not by absolute value)
        eigs = np.linalg.eigvalsh(matrix)
        # find the indices of the two eigenvalues with the smallest absolute values
        idx = np.argpartition(np.abs(eigs), 2)[:2]
        # retrieve and sort them by absolute value
        return sorted(eigs[idx], key=lambda x: abs(x))