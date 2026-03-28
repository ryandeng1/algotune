from typing import Dict, List
import numpy as np
import ot

class Solver:
    """
    EMD solver using the accelerated linear programming routine from POT.
    """
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Parameters
        ----------
        problem : dict
            Dictionary containing:
                * 'source_weights' : 1‑D array of shape (n, ) – weights of the source.
                * 'target_weights' : 1‑D array of shape (m, ) – weights of the target.
                * 'cost_matrix'    : 2‑D array of shape (n, m) – transport cost between every pair.

        Returns
        -------
        dict
            Dictionary with key "transport_plan" whose value is the optimal
            plan G as a list of lists of floats.
        """
        # Ensure inputs are 1‑D float64 and contiguous
        a = np.asarray(problem["source_weights"], dtype=np.float64, order="C")
        b = np.asarray(problem["target_weights"], dtype=np.float64, order="C")
        M = np.asarray(problem["cost_matrix"], dtype=np.float64, order="C")

        # Compute the optimal transport plan using POT's linear programming solver.
        # `check_marginals=False` skips a redundant feasibility test which saves time.
        G = ot.lp.emd(a, b, M, check_marginals=False)

        # Convert to native Python list of lists for the required output format.
        return {"transport_plan": G.tolist()}