import numpy as np
from typing import Any, Dict, List


class Solver:
    """
    Solve the L0 pruning problem:
        min_w ||v - w||^2  s.t.  ||w||_0 <= k
    The optimal solution is obtained by keeping the k components of v with
    the largest absolute values and setting the rest to zero.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                * "v" : list[float] – input vector
                * "k" : int          – sparsity level

        Returns
        -------
        dict
            Dictionary with key "solution" containing the optimal vector.
        """
        v = np.asarray(problem["v"], dtype=np.float64)
        k = int(problem["k"])

        if v.ndim != 1:
            v = v.ravel()

        n = v.size
        if k <= 0:
            return {"solution": [0.0] * n}
        if k >= n:
            return {"solution": v.tolist()}

        # Use argpartition for O(n) average runtime to find k largest abs values
        idx = np.argpartition(np.abs(v), -k)[-k:]
        w = np.zeros_like(v)
        w[idx] = v[idx]

        return {"solution": w.tolist()}
