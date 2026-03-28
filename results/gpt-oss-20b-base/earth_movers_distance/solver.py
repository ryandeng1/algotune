import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Solve the EMD problem using ot.lp.emd.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary with keys 'source_weights', 'target_weights',
            and 'cost_matrix' containing NumPy arrays.

        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary containing the optimal transport plan matrix G
            under the key 'transport_plan'.
        """
        # Retrieve problem data
        a = problem['source_weights']
        b = problem['target_weights']
        M = problem['cost_matrix']

        # Ensure correct memory layout for the cost matrix
        M_cont = np.ascontiguousarray(M, dtype=np.float64)

        # Compute optimal transport plan using underlying C implementation
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)

        return {'transport_plan': G}