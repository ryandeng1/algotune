from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Computes the singular value decomposition of the matrix provided
        in ``problem['matrix']``.
        The matrix is expected to be a NumPy array or a nested list convertible
        to one via `np.asarray`.  The result is returned with lists to
        conform to the problem interface, while all heavy lifting is done
        with NumPy's highly optimised C implementation.

        Parameters
        ----------
        problem : dict
            Dictionary with a single key ``'matrix'``.

        Returns
        -------
        dict
            ``{'U': U, 'S': S, 'V': V}`` where ``U`` and ``V`` are lists of lists
            and ``S`` is a list of singular values.
        """
        A = np.asarray(problem['matrix'])
        U, s, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        V = Vh.T
        return {
            'U': U.tolist(),
            'S': s.tolist(),
            'V': V.tolist()
        }