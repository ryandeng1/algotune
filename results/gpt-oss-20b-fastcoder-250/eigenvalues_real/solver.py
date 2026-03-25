import numpy as np

class Solver:
    def solve(self, problem):
        """Return sorted eigenvalues of a symmetric matrix."""
        # np.linalg.eigh returns eigenvalues sorted ascending
        eigs = np.linalg.eigh(problem)[0]
        # sort descending
        return sorted(eigs, reverse=True)
