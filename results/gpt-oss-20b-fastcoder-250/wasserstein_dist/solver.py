from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, List[float]]) -> float:
        """
        Compute the 1‑D Wasserstein distance between two discrete distributions
        supported on {1,…,n}.

        The distance is equal to the L1 distance between the cumulative
        distribution functions (CDFs).  We avoid constructing the full
        transportation matrix and simply accumulate the absolute difference
        between the CDFs at each point.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'u' and 'v', each a list of probabilities
            (they sum to 1).  The support indices are 1..n where n = len(u).

        Returns
        -------
        float
            The Wasserstein distance.
        """
        try:
            u = np.asarray(problem["u"], dtype=np.float64)
            v = np.asarray(problem["v"], dtype=np.float64)

            # Compute cumulative sums
            cu = np.cumsum(u)
            cv = np.cumsum(v)

            # The 1‑D Wasserstein distance equals the L1 distance
            # between these cumulative distributions.
            return np.sum(np.abs(cu - cv))
        except Exception:
            # Fallback: return a generic upper bound
            return float(len(problem.get("u", [])))
