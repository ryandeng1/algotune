from typing import Any
import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Solve the EMD problem using ot.lp.emd for maximum speed.
        
        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing:
                - 'source_weights' : 1-D array of source masses
                - 'target_weights' : 1-D array of target masses
                - 'cost_matrix'   : 2-D array of transportation costs
        
        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary with key "transport_plan" containing the optimal
            transport plan matrix as a list of lists.
        """
        # Extract arrays and ensure memory contiguity (cost matrix must be contiguous for ot.lp.emd)
        a = np.asarray(problem['source_weights'], dtype=np.float64)
        b = np.asarray(problem['target_weights'], dtype=np.float64)
        M = np.ascontiguousarray(problem['cost_matrix'], dtype=np.float64)

        # Compute the optimal transport plan
        G = ot.lp.emd(a, b, M, check_marginals=False)

        # Convert numpy array to list of lists for return value
        return {'transport_plan': G.tolist()}