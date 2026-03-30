import numpy as np
from typing import Any, Dict, List


class Solver:

    @staticmethod
    def _subproblem_sol(vn: np.ndarray, z: float) -> np.ndarray:
        """
        Compute the optimal solution for the subproblem
        described in the referenced paper.  The implementation
        uses NumPy operations only and avoids explicit Python loops.
        """
        # Work only on the positive part
        mu = np.sort(vn)[::-1]  # descending
        # Cumulative sum of the sorted values
        cum = np.cumsum(mu)
        # Compute the threshold candidates for each rank j
        idx = np.arange(1, len(mu) + 1, dtype=float)
        # candidate thresholds ϑ_j  = (cum_j - z) / j
        theta_cand = (cum - z) / idx
        # The condition is mu_j < theta_j
        mask = mu < theta_cand
        if mask.any():
            j = np.argmax(mask)  # first index where the condition holds
            theta = theta_cand[j]
        else:
            theta = 0.0
        # Final clipped solution
        return np.maximum(vn - theta, 0.0)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the l₁‑pruning problem in O(n log n) time.

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys
            - 'v' : array‑like vector (length n)
            - 'k' : float, the sparsity budget

        Returns
        -------
        dict
            {'solution': [float, …]} where the list has length n.
        """
        v = np.asarray(problem['v'], dtype=float).ravel()
        k = float(problem['k'])

        # Step 1: prune the absolute values
        abs_v = np.abs(v)
        bd = self._subproblem_sol(abs_v, k)

        # Step 2: restore the signs of the original vector
        new_v = bd * np.sign(v)

        # Step 3: return the solution as a plain Python list
        return {'solution': new_v.tolist()}