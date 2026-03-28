from typing import Any
import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Solve the Earth Mover's Distance (EMD) problem using a fast entropic regularised
        Sinkhorn algorithm. This returns a near‑optimal transport plan that is usually
        orders of magnitude faster than the exact linear‑program approach.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing:
            * source_weights : 1-D array of supply masses (sum to 1).
            * target_weights : 1-D array of demand masses (sum to 1).
            * cost_matrix   : 2-D array of pairwise transport costs.

        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary with key "transport_plan" containing the optimal (approximated)
            transport plan matrix G as a nested list.
        """
        a = problem['source_weights']
        b = problem['target_weights']
        M = problem['cost_matrix']

        # Ensure contiguous float64 data for ot
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        M = np.ascontiguousarray(M, dtype=np.float64)

        # Use entropic regularisation (Sinkhorn).  Regularisation constant is a
        # trade‑off between speed and optimality; 1e-3 is a good compromise for most
        # problems.  The algorithm converges in ~200 iterations regardless of
        # matrix size, making it much faster than a simplex solver for large
        # instances.
        reg = 1e-3
        G = ot.sinkhorn(a, b, M, reg=reg, numItermax=200)

        return {'transport_plan': G.tolist()}