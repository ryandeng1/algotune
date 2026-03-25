import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero.
        Return them sorted by absolute value.
        """
        matrix = np.asarray(problem["matrix"], dtype=float)
        # For symmetric matrices, eigvalsh is efficient
        eigs = np.linalg.eigvalsh(matrix)
        # Sort by absolute value and take the two smallest
        return sorted(eigs, key=abs)[:2]
