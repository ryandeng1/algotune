# solver.py
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero.

        Parameters
        ----------
        problem : dict
            Dictionary containing a key 'matrix' with the symmetric matrix
            represented as a nested list of floats.

        Returns
        -------
        list[float]
            Two eigenvalues sorted by absolute value, closest to zero.
        """
        # Convert to numpy array
        mat = np.array(problem["matrix"], dtype=float)

        # Compute eigenvalues (sorted ascending automatically)
        eigs = np.linalg.eigvalsh(mat)

        # Pick the two with smallest absolute value
        idx = np.argsort(np.abs(eigs))
        return [float(eigs[i]) for i in idx[:2]]
